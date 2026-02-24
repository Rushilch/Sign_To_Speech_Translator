import os, cv2, numpy as np, mediapipe as mp
from tqdm import tqdm

DATA_DIR = "data"
OUT = "hybrid_data"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def norm_hand(h):
    h -= h.mean(axis=0)
    s = np.linalg.norm(h, axis=1).max()
    if s > 1e-6:
        h /= s
    return h

def extract(img):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands, \
         mp_face.FaceMesh(static_image_mode=True) as face:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hres = hands.process(rgb)
        fres = face.process(rgb)

        if not fres.multi_face_landmarks or not hres.multi_hand_landmarks:
            return None

        pts = []
        for lm in hres.multi_hand_landmarks:
            p = np.array([[x.x, x.y] for x in lm.landmark])
            pts.append(p)

        if len(pts) == 1:
            pts = pts * 2

        pts.sort(key=lambda p: p[:,0].mean())
        left, right = norm_hand(pts[0]), norm_hand(pts[1])

        nose = fres.multi_face_landmarks[0].landmark[1]
        face = np.array([nose.x, nose.y])

        lh_c, rh_c = left.mean(axis=0), right.mean(axis=0)

        static = np.concatenate([
            left.flatten(), right.flatten(),
            [np.linalg.norm(lh_c - face)],
            [np.linalg.norm(rh_c - face)]
        ])

        motion = np.zeros(8)   # placeholder for live app
        return np.concatenate([static, motion])

def main():
    os.makedirs(OUT, exist_ok=True)
    labels = []

    for cls in sorted(os.listdir(DATA_DIR)):
        src = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(src):
            continue

        labels.append(cls)
        cls_out = os.path.join(OUT, cls)
        os.makedirs(cls_out, exist_ok=True)

        i = 0
        for f in tqdm(os.listdir(src)):
            img = cv2.imread(os.path.join(src, f))
            if img is None:
                continue
            feat = extract(img)
            if feat is None:
                continue
            np.save(os.path.join(cls_out, f"{i}.npy"), feat)
            i += 1

    with open("hybrid_labels.txt", "w") as f:
        f.write("\n".join(labels))

    print("✅ Hybrid static dataset built")

if __name__ == "__main__":
    main()
