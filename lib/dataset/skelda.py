from __future__ import absolute_import, division, print_function

import json
import logging

import numpy as np
import tqdm
from dataset.JointsDataset import JointsDataset
from skelda import evals, utils_pose

logger = logging.getLogger(__name__)


# ==================================================================================================

dataset_use = "human36m"
# dataset_use = "panoptic"
# dataset_use = "mvor"
# dataset_use = "shelf"
# dataset_use = "campus"
# dataset_use = "ikeaasm"
# dataset_use = "tsinghua"
datasets = {
    "panoptic": {
        "path": "/datasets/panoptic/skelda/test.json",
        "cams": ["00_03", "00_06", "00_12", "00_13", "00_23"],
        "take_interval": 3,
        "use_scenes": ["160906_pizza1", "160422_haggling1", "160906_ian5"],
    },
    "human36m": {
        "path": "/datasets/human36m/skelda/pose_test.json",
        "take_interval": 5,
    },
    "mvor": {
        "path": "/datasets/mvor/skelda/all.json",
        "take_interval": 1,
    },
    "campus": {
        "path": "/datasets/campus/skelda/test.json",
        "take_interval": 1,
    },
    "shelf": {
        "path": "/datasets/shelf/skelda/test.json",
        "take_interval": 1,
    },
    "ikeaasm": {
        "path": "/datasets/ikeaasm/skelda/test.json",
        "take_interval": 2,
    },
    "tsinghua": {
        "path": "/datasets/tsinghua/skelda/test.json",
        "take_interval": 3,
    },
}

joint_names_3d = [
    "shoulder_middle",
    "nose",
    "hip_middle",
    "shoulder_left",
    "elbow_left",
    "wrist_left",
    "hip_left",
    "knee_left",
    "ankle_left",
    "shoulder_right",
    "elbow_right",
    "wrist_right",
    "hip_right",
    "knee_right",
    "ankle_right",
]

eval_joints = [
    "nose",
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
]

# ==================================================================================================


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# ==================================================================================================


def load_labels(dataset: dict):
    """Load labels by dataset description"""

    if "panoptic" in dataset:
        labels = load_json(dataset["panoptic"]["path"])
        labels = [lb for i, lb in enumerate(labels) if i % 1500 < 90]

        # Filter by maximum number of persons
        labels = [l for l in labels if len(l["bodies3D"]) <= 10]

        # Filter scenes
        if "use_scenes" in dataset["panoptic"]:
            labels = [
                l for l in labels if l["scene"] in dataset["panoptic"]["use_scenes"]
            ]

        # Filter cameras
        if not "cameras_depth" in labels[0]:
            for label in labels:
                for i, cam in reversed(list(enumerate(label["cameras"]))):
                    if cam["name"] not in dataset["panoptic"]["cams"]:
                        label["cameras"].pop(i)
                        label["imgpaths"].pop(i)

    elif "human36m" in dataset:
        labels = load_json(dataset["human36m"]["path"])
        labels = [lb for i, lb in enumerate(labels) if i % 9000 < 150]

        for label in labels:
            label.pop("action")
            label.pop("frame")

    elif "mvor" in dataset:
        labels = load_json(dataset["mvor"]["path"])

        # Rename keys
        for label in labels:
            label["cameras_color"] = label["cameras"]
            label["imgpaths_color"] = label["imgpaths"]

            # Use "head" label for "nose" detections
            label["joints"][label["joints"].index("head")] = "nose"

            # Lift poses and cams up, that the bottom of the room is at zero level
            # Else the dataset pipeline fails
            bodies3D = np.array(label["bodies3D"])
            bodies3D += [0, 0, 1.0, 0]
            label["bodies3D"] = bodies3D.tolist()
            for cam in label["cameras"]:
                T = np.array(cam["T"])
                T[2][0] += 1.0
                cam["T"] = T.tolist()

    elif "ikeaasm" in dataset:
        labels = load_json(dataset["ikeaasm"]["path"])
        labels = [lb for i, lb in enumerate(labels) if i % 300 < 72]

        # Lift poses and cams up, that the bottom of the room is at zero level
        # Else the dataset pipeline fails
        for label in labels:
            bodies3D = np.array(label["bodies3D"])
            bodies3D += [0, 0, 1.2, 0]
            label["bodies3D"] = bodies3D.tolist()
            for cam in label["cameras"]:
                T = np.array(cam["T"])
                T[2][0] += 1.2
                cam["T"] = T.tolist()

    elif "shelf" in dataset:
        labels = load_json(dataset["shelf"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]
        for label in labels:
            label["scene"] = "main"

            # Use "head" label for "nose" detections
            label["joints"][label["joints"].index("head")] = "nose"

    elif "campus" in dataset:
        labels = load_json(dataset["campus"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]
        for label in labels:
            label["scene"] = "main"

            # Use "head" label for "nose" detections
            label["joints"][label["joints"].index("head")] = "nose"

    elif "tsinghua" in dataset:
        labels = load_json(dataset["tsinghua"]["path"])
        labels = [lb for lb in labels if "test" in lb["splits"]]
        labels = [lb for i, lb in enumerate(labels) if i % 800 < 90]

        for label in labels:
            label["scene"] = label["seq"]
            label["bodyids"] = list(range(len(label["bodies3D"])))

    else:
        raise ValueError("Dataset not available")

    # Optionally drop samples to speed up train/eval
    if "take_interval" in dataset:
        take_interval = dataset["take_interval"]
        if take_interval > 1:
            labels = [l for i, l in enumerate(labels) if i % take_interval == 0]

    # Filter joints
    fj_func = lambda x: utils_pose.filter_joints_3d(x, joint_names_3d)
    labels = list(map(fj_func, labels))

    return labels


# ==================================================================================================


class Skelda(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)

        self.num_joints = len(joint_names_3d)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOT_JOINT_ID

        self.has_evaluate_function = True
        self.transform = transform

        print("Loading labels ...")
        if is_train == "train":
            labels = []

        else:
            ds = datasets[dataset_use]
            labels = load_labels(
                {dataset_use: ds, "take_interval": ds["take_interval"]}
            )
        self.labels = labels

        # Print a dataset sample for debugging
        print(labels[0])

        self.has_views = len(labels[0]["cameras"])
        self.num_views = cfg.DATASET.CAMERA_NUM
        print(self.num_views, self.has_views)

        self._get_db()
        self.db_size = len(self.db)
        self.cameras = self._get_cam()
        print(len(self.labels), self.db_size)

    def _get_db(self):
        db = []

        for label in tqdm.tqdm(self.labels):
            all_poses_3d = []
            all_poses_vis_3d = []

            for body in label["bodies3D"]:
                pose = np.array(body)

                pose3d = pose[:, 0:3] * 1000
                vis3d = pose[:, -1] > 0.1

                all_poses_3d.append(pose3d)
                all_poses_vis_3d.append(vis3d)

            item = {
                "all_image_path": label["imgpaths"],
                "seq": label["scene"],
                "idx": label["id"],
                "joints_3d": all_poses_3d,
                "joints_3d_vis": all_poses_vis_3d,
            }
            db.append(item)

        self.db = db
        super()._rebuild_db()
        logger.info(
            "=> {} images from {} views loaded".format(len(self.db), self.num_views)
        )
        return

    def _get_cam(self):
        cameras = {}
        for label in tqdm.tqdm(self.labels):
            if label["scene"] not in cameras:
                cameras[label["scene"]] = [{} for _ in range(self.has_views)]

            for i, cam in enumerate(label["cameras"]):
                our_cam = {}
                our_cam["R"] = np.array(cam["R"])
                our_cam["T"] = np.array(cam["T"]) * 1000
                our_cam["fx"] = np.array(cam["K"])[0, 0]
                our_cam["fy"] = np.array(cam["K"])[1, 1]
                our_cam["cx"] = np.array(cam["K"])[0, 2]
                our_cam["cy"] = np.array(cam["K"])[1, 2]
                our_cam["k"] = np.array(cam["DC"])[[0, 1, 4]].reshape(3, 1)
                our_cam["p"] = np.array(cam["DC"])[[2, 3]].reshape(2, 1)
                cameras[label["scene"]][i] = our_cam

        return cameras

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap

    def __len__(self):
        return self.db_size

    # ==============================================================================================

    def evaluate(self, preds):
        global joint_names_3d

        all_poses = preds.detach().cpu().numpy()
        all_ids = [r["id"] for r in self.labels]

        filtered_poses = []
        scale = [1000.0, 1000.0, 1000.0, 1.0]
        for poses in all_poses:
            new_poses = []
            for pose in poses:
                if pose[self.root_id, 3] >= 0:
                    pose = np.array(pose)[:, [0, 1, 2, 4]] / scale

                    new_poses.append(pose)
            filtered_poses.append(new_poses)
        all_poses = filtered_poses

        res = evals.mpjpe.run_eval(
            self.labels,
            all_poses,
            all_ids,
            joint_names_net=joint_names_3d,
            joint_names_use=eval_joints,
            save_error_imgs="",
            pred_imgpaths=[],
        )
        _ = evals.pcp.run_eval(
            self.labels,
            all_poses,
            all_ids,
            joint_names_net=joint_names_3d,
            joint_names_use=eval_joints,
            replace_head_with_nose=True,
        )

        if "mpjpe" in res:
            metric = np.mean(
                [v for k, v in res["mpjpe"].items() if k.startswith("ap-")]
            )
        else:
            metric = 0

        return metric, ""
