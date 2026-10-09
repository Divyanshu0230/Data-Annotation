# F05 dense 4K — full project record

This file is the complete description of the labeling project in this repository. It matches `annotate.py`, `labels.json`, and `note.txt`.

Author: Divyanshu  
Repository: https://github.com/Divyanshu0230/Data-Annotation  
Clip id in the label file: `F05_dense_4k`

## 1. What the project is

One continuous drone shot over a city intersection was labeled for two training tasks at once.

- Detection: where each object is in a frame.
- Multi-object tracking: which physical object is which from frame to frame.

Both tasks come from one annotation. The field that makes tracking possible is `instance_id`. The same car or person keeps the same id in every frame where it appears, until it leaves the image.

The clip is 20 seconds, extracted at the native 60 fps. There are 1,199 frames. None were skipped, sampled, resized, renamed, or re-encoded. The JPEG files are 3840 × 2160. They are not stored in this git repository. On the machine that produced the labels they lived in `/Users/divyanshu/Desktop/f05_frames`, named `frame_00001.jpg` through `frame_01199.jpg`. `sequence` in the JSON is that number: `frame_00007.jpg` is sequence 7.

Hand-drawing every box was not realistic. A busy frame has on the order of 300 objects, and the assignment itself says not to click every box. The method used here is a detector on every frame, then a tracker. At 60 fps an object only moves a few pixels between frames, so a detection on every frame is stricter than drawing a keyframe every 10–30 frames and interpolating the rest. Interpolation in this project is only a patch for a short detector miss.

## 2. Counts in labels.json

| Item | Value |
|---|---|
| Frames | 1,199, sequences 1 through 1199, each exactly once, ascending |
| Image size | `[3840, 2160]` |
| fps | 60 |
| frame_count | 1199 |
| annotator | `divyanshu` |
| Boxes | 401,510 |
| Mean boxes per frame | about 335 |
| Empty frames | 0 |
| Instance ids | 1 through 1786, one shared pool |
| Vehicle tracks | 1,170 (`class_id` 0) |
| Person tracks | 616 (`class_id` 1) |
| Class flips on one id | 0 |
| Duplicate instance id inside one frame | 0 |
| Vehicle boxes | 313,054 |
| Person boxes | 88,456 |
| Occlusion 0 / 1 / 2 | 347,977 / 48,119 / 5,414 |
| Truncated boxes | 7,891 |

The header is fixed by the assignment: clip, image size, fps, and frame count are copied as specified.

## 3. Classes

Only two classes are written.

| class_id | Name | What goes in it |
|---|---|---|
| 0 | vehicle | car, truck, bus, van, auto-rickshaw, motorcycle, scooter, bicycle |
| 1 | person | pedestrians, and riders whose body is clearly separate from the vehicle |

Anything else is not annotated. Reflections, shadows, and pictures of objects are not objects. A car painted on a billboard is not a car. A crowd is never one box. Each person gets their own box and id.

The detector is a YOLO11-L checkpoint fine-tuned on VisDrone (`models/yolo11l-visdrone.pt`). VisDrone has more classes than this task. They are collapsed as follows.

| VisDrone id | VisDrone name | Written as |
|---|---|---|
| 0 | pedestrian | person, class_id 1 |
| 1 | people (crowd box) | dropped |
| 2 | bicycle | vehicle, class_id 0 |
| 3 | car | vehicle |
| 4 | van | vehicle |
| 5 | truck | vehicle |
| 6 | tricycle | vehicle |
| 7 | awning-tricycle | vehicle (auto-rickshaw style) |
| 8 | bus | vehicle |
| 9 | motor | vehicle |
| 10 | ignored if it appears | dropped |

The detector is asked only for classes `[0, 2, 3, 4, 5, 6, 7, 8, 9]`.

Two-wheel group used by the rider rule: bicycle, tricycle, awning-tricycle, motor (`{2, 6, 7, 9}`).  
Four-wheel group: car, van, truck, bus (`{3, 4, 5, 8}`).

## 4. What one box contains

```json
{
  "box": [723.4, 2.7, 21.0, 18.6],
  "class_id": 0,
  "instance_id": 1,
  "occlusion": 0,
  "truncated": false
}
```

| Field | Meaning |
|---|---|
| box | `[x, y, w, h]` in absolute pixels. Origin is the top-left. `x, y` is the top-left of the box. One decimal place. |
| class_id | 0 or 1 |
| instance_id | Identity of this physical object across the whole clip |
| occlusion | 0 fully visible, 1 partly hidden (up to about half), 2 mostly hidden (more than half) |
| truncated | true when the object is cut by the image edge |

The assignment asks for a tight box that touches the outermost pixels, with no padding. These boxes are the detector boxes after a light median smooth. They are tight on ordinary cars. Trucks, cement mixers, and the tanker are looser, and in a dense queue two boxes sometimes touch.

Truncated objects are clipped to the image. Only the visible part is stored, and `truncated` is true when the box touches the border (margin about 2 pixels).

Partly occluded objects keep the detected box. The hidden part was not drawn out to a guessed full extent. Completely hidden objects have no box on that frame.

Occlusion is computed from overlap with a nearer box. In this oblique aerial view, nearer means the other box has a lower bottom edge (its `y + h` is at least 4 pixels greater). Cover under about 18% is occlusion 0. Cover from that up to about 55% is occlusion 1. Cover at or above about 55% is occlusion 2.

## 5. Instance id rules

- Ids start at 1 and increase in order of first appearance. If two objects start on the same frame, the one higher in the image (smaller y) comes first, then the one further left.
- Vehicles and people share one pool. There is never a vehicle 3 and a person 3.
- An id is unique in a frame.
- Leaving the frame retires the id. The number is never used again. If the same person walks back in later, they get a new id. The last visible boxes are normally `truncated: true`.
- Hiding inside the frame does not retire the id. Frames where the object is fully hidden have no record. When it reappears near the camera-compensated position, the same id continues.
- If the reappearance is too far from that position, the track is split and the later piece gets a new id. A wrong merge is worse than a split.
- Parked cars and people who are standing still keep one id for as long as they stay in the image.

Worked example from the assignment, which this file follows:

| Frames | Event | Result |
|---|---|---|
| 1–120 | person walks in from the left | instance_id 1, class 1 |
| 30–200 | car enters from the left | instance_id 2, class 0 |
| 121 | person exits the right edge | id 1 retired |
| 150–160 | car fully hidden behind a bus | no record for id 2 in 150–160; resumes as 2 at 161 |
| 300 | same person re-enters from the right | new instance_id 3, class 1 |

In this implementation the "still the same object" window while hidden inside the image is 48 frames, which is 0.8 seconds. On the border it is 8 frames. If less than about 28% of the predicted box is still inside the image, the id ends immediately.

## 6. Rider and occupant rule

The two-wheeler is always a vehicle.

A person box is removed when it is mostly inside a four-wheeler (car, van, truck, bus): if more than half of the person box overlaps that vehicle, the person is treated as an occupant and dropped.

A person overlapping a two-wheeler by more than 60% is kept only when the top of the person box sits clearly above the vehicle, by at least 22% of the vehicle height. That is the "body reads as separate" test. Otherwise the person and the vehicle are one blob and only the vehicle is kept.

People inside a vehicle who are not a separate figure are not annotated.

## 7. Pipeline

```
frames (3840x2160 JPEG, untouched)
        |
        v
annotate.py detect
        |  YOLO11-L, imgsz 1920, boxes mapped back to 3840x2160
        |  class filter, size filter, NMS, occupant filter
        |  homography from previous frame to this frame
        v
detections.npz
        |
        v
annotate.py track
        |  warp by homography, add object velocity
        |  match, spawn, retire, split, smooth, fill short gaps
        |  occlusion and truncated flags
        v
labels.json
```

Commands:

```bash
python3 annotate.py detect    # writes detections.npz; needs weights and frames
python3 annotate.py track     # writes labels.json from detections.npz
python3 annotate.py qa        # draws sample overlays into qa/
python3 annotate.py all
```

`track` validates the JSON before it exits: header values, frames 1–1199 once each, class 0 or 1, occlusion 0/1/2, positive box size, box inside the image, no duplicate id in a frame, ids are the contiguous set 1..N.

Dependencies are in `requirements.txt`: numpy, opencv-python, scipy, torch, torchvision, ultralytics. Detection in this run used Apple MPS (`device="mps"`).

Detection of all 1,199 frames took about 13 minutes. Linking took under a minute. Checking samples and deciding the edge cases took longer than the compute. The whole task was a few hours.

## 8. Detection, frame by frame

`detect()` loads `models/yolo11l-visdrone.pt`.

For every frame it calls YOLO with:

| Argument | Value |
|---|---|
| imgsz | 1920 |
| conf | 0.15 |
| iou | 0.5 |
| max_det | 1200 |
| device | mps |
| classes | 0, 2, 3, 4, 5, 6, 7, 8, 9 |

The JPEG is read at full resolution and is never written back. Ultralytics scales the image internally to an inference size of 1920 and returns boxes in the original pixel coordinates.

`prepare_dets` then:

- Converts xyxy to xywh.
- Drops boxes thinner than 4 px or shorter than 6 px.
- Drops class 1 and class 10.
- Drops extreme aspect ratios (width/height or height/width of 8 or more).
- Runs NMS separately: people at IoU 0.40, vehicles at IoU 0.45.
- Applies `filter_occupants` as described in section 6.

The file can resume. If `detections.npz` exists and `n_done` is less than 1199, detection continues from the next frame. If `n_done` is already 1199, detect exits without redoing the clip.

On this clip the raw stored detections were 396,870 boxes before the tracker score gates, about 331 per frame, ranging from 188 to 487. Median confidence was about 0.64. Median box size was about 31 × 32 pixels. The largest box was about 182 × 165. That is a truck or mixer, not a group box around a whole junction.

## 9. Camera motion

The drone moves. Between most frames the shift is about 2 pixels. Around frames 1021–1038 it is about 43 pixels per frame, a real pan, not a glitch. Scale stays within about 0.2% of 1.

`compute_homography` maps previous-frame pixels onto the current frame.

- Both frames are resized to a width of 960 for the flow, area interpolation, grayscale.
- Up to 700 Shi-Tomasi corners, quality 0.01, min distance 7, block size 7.
- Pyramidal Lucas-Kanade, window 21, 3 levels.
- Corners that fall inside a detection, expanded by 4 pixels, are removed so traffic does not pull the camera estimate. At least 40 points must remain or the mask is skipped.
- Homography is RANSAC with a 3 pixel threshold. At least 30 inliers are required.
- Rejected, and replaced by the identity, if the linear part is more than about 0.06 away from a pure shift, or the full-resolution translation is more than 60 pixels.
- The 960-pixel homography is scaled back to 3840 × 2160.

In the actual run, every frame accepted a homography. The identity fallback count was 0.

`detections.npz` stores:

- `data`: float32 rows `[x, y, w, h, confidence, visdrone_class]`
- `offsets`: length 1200, start index of each frame in `data`
- `H`: shape `(1199, 3, 3)`, where `H[k]` maps frame k onto frame k+1 for k ≥ 1. `H[0]` is unused identity.
- `n_done`: 1199

Tracking reads this file. It does not run the network again.

## 10. Tracking

For each frame the score gate keeps:

- person confidence ≥ 0.18
- four-wheel ≥ 0.20
- two-wheel ≥ 0.22

Live tracks are predicted by warping the last box with `H`, then adding velocity `(vx, vy)`. Velocity is the residual motion after the warp, smoothed as `0.45 * old + 0.55 * new`. On a miss, velocity is multiplied by 0.55.

Association is the Hungarian algorithm on a cost of `(1 - IoU) + 0.10 * (center distance / gate)`.

A pair is allowed only when:

- The superclass matches. A person is never matched to a vehicle.
- Area ratio is between 0.34 and 2.9.
- IoU is at least 0.12, or IoU is at least 0.02 and the centers are inside the gate and the area ratio is between 0.45 and 2.3.
- Center distance is at most `max(2.2 * gate, 28)` pixels.
- Gate is `max(18, 0.50 * max(width, height))` of the predicted box.

High-confidence detections (score ≥ 0.34) are matched first. Unmatched tracks then try the remaining lower-score detections. This is the ByteTrack pattern: a weak box may continue a track, but section 11 stops it from creating one.

A matched track stores the detection as the box for that frame, resets the miss counter, and becomes confirmed after 3 hits or a score of at least 0.55.

An unmatched track:

- If less than about 28% of the predicted box remains inside the image, the id is retired (kept only if it was confirmed and had at least 2 real observations).
- If the prediction touches the border, it may coast 8 frames. Otherwise it may coast 48 frames. During the coast, no box is written.
- An unconfirmed track is dropped after 2 misses.

## 11. Starting a new id

An unmatched detection starts a track only if `spawn_ok` passes.

| Kind | Inside the image | Touching the border |
|---|---|---|
| person | confidence ≥ 0.27 | ≥ 0.22 |
| two-wheel | ≥ 0.30 | ≥ 0.25 |
| other vehicle | ≥ 0.26 | ≥ 0.22 |

Border objects are allowed a lower score because they are entering or leaving and are partly cut off.

## 12. After all frames

1. Split. For each pair of successive real observations, the earlier box is warped through the homographies to the later frame. If the center moved more than the limit, the track is cut and the later piece becomes its own id.
   - For a gap of 8 frames or fewer, the limit is `(14 + 0.35 * size) * gap` pixels, where size is the larger side of the box (at least 8).
   - For a longer gap, the limit is `1.05 * size + 2.0 * gap`.
   - A piece with only one observation is dropped.
2. Keep or drop. A track needs at least 2 hits. It is kept if it has 5 or more observations, or max confidence ≥ 0.55, or it touched the border with max confidence ≥ 0.38, or max confidence ≥ 0.36 and at least 3 observations. One-frame specks are removed.
3. Smooth. Inside a contiguous run of frames (no missing frame), the center is a 3-frame median and width and height are a 5-frame median. Runs shorter than 4 frames are not smoothed. A gap is never included in the median, so a box cannot be pulled halfway between two places.
4. Fill. A hole of 1 to 8 frames between two real observations is linear interpolation of `x, y, w, h`. A longer hole stays empty. The id does not change across that hole if the split step left them as one track.
5. Drop a filled box if a nearer box covers about 65% or more of it. That frame is treated as fully hidden.
6. Drop a box whose clipped width is under 3 px or height under 4 px, or whose visible fraction is under 0.15.
7. Assign ids 1..N in order of start frame, then top-to-bottom, then left-to-right.
8. Write compact JSON. Round box numbers to 1 decimal.

## 13. Edge cases and what was decided

Riders. Vehicle always. Person only when the body extends above the two-wheeler as in section 6. One blob means vehicle only.

Occupants. A person box covered more than half by a car, van, truck, or bus is removed.

Groups. The crowd class is never written. Each detected person who survives the filters gets their own box and id.

Auto-rickshaws. VisDrone awning-tricycle and tricycle are vehicles.

Billboards and signs. A pictured car is not a car. In the frames that were checked, the model was not boxing signage as vehicles.

Small and distant objects. They are kept when the detector holds them across enough frames to pass section 12. A detection that appears once or twice and never reaches the keep rule is dropped. The assignment asked for small objects, so the score gates for extending a track are lower than the gates for starting a new id.

Poles, shadows, rooftop clutter. Some of these look like a person or a motorcycle. Tracks that did not stay steady were dropped. A few that the detector held for a long run were kept. Those are the least certain boxes in the file.

Parked vehicles. They are not special-cased as "static, so ignore." They keep one id across the frames they are visible, the same as moving traffic.

Cement mixers and the tanker. Real vehicles, class 0. Their boxes are larger and looser than the cars around them. They were left as detected.

Dense queue. Neighbouring cars sometimes have boxes that touch or overlap slightly. NMS at 0.45 removes strong duplicates. A small number of pairs still overlap. They were not hand-separated.

Occlusion. Estimated from overlap with a nearer box, not from painting the hidden outline. A partly hidden object keeps the visible detection. A fully hidden stretch has no box. An interpolated box that would be almost completely covered is omitted.

Identity checks that were actually done. Random frames were opened, including consecutive frames such as 600 and 601, and individual tracks were followed. One white sedan kept a single id from early in its life until it left the image edge, where the last box is truncated. Frame-to-frame motion after removing camera motion is about 2% of the box size at the median and about 15% at the 99th percentile. That is the check that ids are not hopping to a neighbour every frame.

No mean average precision was measured. No second detector was ensembled. No count of "hand-fixed boxes" exists, because the boxes were not hand-drawn.

## 14. Label file shape

```json
{
  "clip": "F05_dense_4k",
  "image_size": [3840, 2160],
  "fps": 60,
  "frame_count": 1199,
  "annotator": "divyanshu",
  "frames": [
    {
      "sequence": 1,
      "detections": [
        {
          "box": [723.4, 2.7, 21.0, 18.6],
          "class_id": 0,
          "instance_id": 1,
          "occlusion": 0,
          "truncated": false
        }
      ]
    }
  ]
}
```

The file is one compact JSON line, about 37 MB. Google Drive and GitHub show it as a long line. It parses. A frame with nothing in it would still be present with `"detections": []`. This clip has no empty frame.

`qa/` is not in git. `python3 annotate.py qa` writes overlays for frames 1, 30, 120, 300, 600, 601, 900, and 1199. Green is a vehicle, yellow is a person, orange is occluded, blue is truncated. The overlay is scaled to 1440 × 810 for viewing. The labels themselves stay at full resolution.

## 15. Files

| File | Role |
|---|---|
| annotate.py | detect, track, validate, qa |
| detections.npz | per-frame boxes and homographies |
| labels.json | the annotation deliverable |
| models/yolo11l-visdrone.pt | YOLO11-L VisDrone weights, about 49 MB |
| note.txt | the short annotation note submitted with the labels |
| requirements.txt | Python packages |
| README.md | shorter architecture description |
| PROJECT.md | this record |

To rebuild labels, point `FRAMES` at the top of `annotate.py` to the JPEG folder, keep the weights at `models/yolo11l-visdrone.pt`, run `detect`, then `track`. `track` alone is enough when `detections.npz` is already complete.

## 16. Known limits

- Inference at 1920 on a 3840-wide frame misses some very small distant bikes and people that a tiled or larger pass would catch. 1920 was used so the full clip finished in about 13 minutes.
- Truck, mixer, and tanker boxes are looser than car boxes.
- Occlusion is an overlap estimate. The box is not extended to a guessed full physical extent.
- A hide longer than 0.8 seconds, or a reappearance far from the predicted position, becomes a new id even if a human might have joined them.
- A small number of steady false detections on dark poles or rooftops can survive because persistence was preferred over dropping a real pedestrian.
- The homography is a single plane. It is a good model for this slow aerial pan. It is not a 3D scene model.
