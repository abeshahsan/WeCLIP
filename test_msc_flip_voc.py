import argparse
import os
import sys
import warnings

sys.path.append(".")
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from WeCLIP_model.model_attn_aff_voc import WeCLIP
import imageio.v2 as imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default="configs/voc_attn_reg.yaml", type=str, help="config"
)
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")  # val
parser.add_argument(
    "--model_path",
    default="/your/path/WeCLIP/WeCLIP_model_iter_30000.pth",
    type=str,
    help="model_path",
)


def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds, cams = [], [], [], []

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False
    )

    # with torch.no_grad(), torch.cuda.device(0):
    model.cuda(0)
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    for idx, data in tqdm(
        enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="
    ):
        num += 1

        name, inputs, labels, cls_labels = data
        names = name + name

        inputs = inputs.cuda()
        labels = labels.cuda()

        #######
        # resize long side to 512

        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h, w)
        _h, _w = int(h * ratio), int(w * ratio)
        inputs = F.interpolate(
            inputs, size=(_h, _w), mode="bilinear", align_corners=False
        )

        #######

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_cat, cam, attn_loss = model(inputs_cat, names, mode="train")

        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, h, w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(
                    inputs, scale_factor=s, mode="bilinear", align_corners=False
                )
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_cat, cam_cat, attn_loss = model(inputs_cat, names, mode="val")

                _segs_cat = F.interpolate(
                    segs_cat, size=(h, w), mode="bilinear", align_corners=False
                )
                _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(
            segs, size=labels.shape[1:], mode="bilinear", align_corners=False
        )
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(
            msc_segs, size=labels.shape[1:], mode="bilinear", align_corners=False
        )
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))

        if num % 100 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(
                _gts, _msc_preds, _msc_preds_hist
            )
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []

        np.save(
            args.work_dir + "/logit/" + name[0] + ".npy",
            {
                "segs": segs.detach().cpu().numpy(),
                "msc_segs": msc_segs.detach().cpu().numpy(),
            },
        )

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(config.dataset.name_list_dir)
    with open(txt_name) as f:
        name_list = [x for x in f.read().split("\n") if x]

    images_path = os.path.join(
        config.dataset.root_dir,
        "JPEGImages",
    )
    labels_path = os.path.join(config.dataset.root_dir, "SegmentationClassAug")

    post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=3,  # 3
        pos_w=3,  # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,  # 5
        bi_w=4,  # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit["msc_segs"]

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)  # [None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(
            os.path.join(args.work_dir, "prediction", name + ".png"),
            np.squeeze(pred).astype(np.uint8),
        )
        imageio.imsave(
            os.path.join(args.work_dir, "prediction_cmap", name + ".png"),
            encode_cmap(np.squeeze(pred)).astype(np.uint8),
        )
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))]
    )

    preds, gts = zip(*results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores(gts, preds, hist, 21)

    print(score)

    return True


def main(cfg):

    cfg.dataset.root_dir = "H:/archive/VOC2012"
    cfg.clip_init.clip_pretrain_path = os.path.abspath("checkpoint/ViT-B-16.pt")

    # Load image names from the name list file
    name_list_file = os.path.abspath(f"datasets/voc/val.txt")
    with open(name_list_file, "r") as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(image_names)} images to process")

    # Initialize WeCLIP model once
    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device="cuda",
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=os.path.abspath(f"datasets/voc"),
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    # Load pre-trained weights (uncomment if you have a trained model)
    checkpoint = torch.load(
        "F:/UNI_STUFF/Thesis/New folder/WeCLIP/WeCLIP_OG_model_iter_30000.pth",
        map_location="cpu",
        weights_only=False,
    )
    trained_state_dict = checkpoint["model_state_dict"]
    WeCLIP_model.load_state_dict(state_dict=trained_state_dict, strict=False)

    WeCLIP_model.cuda(0)
    WeCLIP_model.eval()


    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(model=WeCLIP_model, dataset=val_dataset, test_scales=[1, 0.75])
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    # # Process all images one by one with progress bar
    # for image_name in tqdm(image_names, desc="Processing images", ncols=100):
    #     try:
    #         # Load image directly
    #         image_path = os.path.join(
    #             cfg.dataset.root_dir, "JPEGImages", f"{image_name}.jpg"
    #         )
    #         image = imageio.imread(image_path)

    #         # Convert to tensor and normalize (following standard ImageNet normalization)
    #         image = image.astype(np.float32) / 255.0
    #         # Apply ImageNet normalization
    #         mean = np.array([0.485, 0.456, 0.406])
    #         std = np.array([0.229, 0.224, 0.225])
    #         image = (image - mean) / std

    #         # Convert to CHW format and add batch dimension
    #         image = (
    #             torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    #         )

    #         # Resize image for processing (as done in validate function)
    #         _, _, h, w = image.shape
    #         ratio = args.resize_long / max(h, w)
    #         _h, _w = int(h * ratio), int(w * ratio)
    #         image = F.interpolate(
    #             image, size=(_h, _w), mode="bilinear", align_corners=False
    #         )

    #         # Forward pass through model
    #         annot_image_path = os.path.join(
    #             cfg.dataset.root_dir, "SegmentationClassAug", f"{image_name}.png"
    #         )

    #         segs_cat, cam, attn_loss = WeCLIP_model(
    #             image, [annot_image_path], mode="train"
    #         )

    #         if cam is not None:
    #             # Take the first CAM (from original image, not flipped)
    #             cam_original = cam[0]  # Shape: [H, W] for refined cam labels

    #             # Load ground truth to get target dimensions
    #             gt_path = os.path.join(
    #                 cfg.dataset.root_dir, "SegmentationClassAug", f"{image_name}.png"
    #             )
    #             gt_image = imageio.imread(gt_path)
    #             gt_h, gt_w = gt_image.shape[:2]

    #             # Resize CAM to match ground truth dimensions
    #             cam_tensor = (
    #                 cam_original.unsqueeze(0).unsqueeze(0).float()
    #             )  # Add batch and channel dims
    #             cam_resized = F.interpolate(
    #                 cam_tensor,
    #                 size=(gt_h, gt_w),
    #                 mode="nearest",  # Use nearest for label interpolation
    #             )
    #             cam_resized = cam_resized.squeeze(0).squeeze(
    #                 0
    #             )  # Remove batch and channel dims

    #             # Convert CAM to numpy
    #             cam_np = cam_resized.cpu().numpy().astype(np.uint8)

    #             # Save as grayscale image
    #             output_path = f"predictions/cam_refined_{image_name}.png"
    #             imageio.imsave(output_path, cam_np)

    #             # print(f"Saved {image_name}: GT shape {gt_image.shape}, CAM shape {cam_np.shape}")

    #     except Exception as e:
    #         print(f"\nError processing {image_name}: {str(e)}")
    #         continue

    print(f"\nCompleted processing all {len(image_names)} images")
    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # print(cfg)
    # print(args)

    # args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)  # For CAM outputs

    main(cfg=cfg)

    # validate
