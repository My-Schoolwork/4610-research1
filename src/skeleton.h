// skeleton.h - Hierarchical articulated skeleton and the procedural walk-cycle
// driver for the Cubism penguin model.
//
// Skeleton model
// --------------
// Each Joint owns a local transform T_local(t) = T(offset) * R(euler(t)) where
// the offset is the pose-space translation from the parent's origin and
// R(euler) is built from per-axis Euler rotations (extrinsic Z * Y * X in our
// convention).  The world transform is the usual left-multiplicative chain
//
//     T_world[i] = T_world[parent(i)] * T_local[i]
//
// as described in Parent, "Computer Animation: Algorithms and Techniques",
// 3rd ed., Morgan Kaufmann, 2012, Ch. 5, and in the OpenGL.org tutorial
// "Hierarchical models" (Paroj 2012).  We compute the entire chain in
// pose-space then feed the final per-mesh matrices to the software rasteriser.
//
// Walk-cycle driver
// -----------------
// The gait is procedural: every DOF is a closed-form function of a phase
// angle phi = 2*pi * f * t, where f is the step frequency.  This follows the
// "cosine-family" controller used extensively in real-time games and
// biomechanics models (Bruderlin & Calvert, "Goal-Directed, Dynamic Animation
// of Human Walking", SIGGRAPH '89; Sun & Metaxas, "Automating Gait Generation",
// SIGGRAPH '01).
//
// Penguin-specific touches, grounded in Griffin & Kram (Nature, 2000) and
// Kurz et al. (J. Theor. Biol., 2008):
//   * Lateral roll of the torso ("waddle") ~= 12 degrees, out of phase with
//     step so that the supporting foot lies directly under the centre of
//     mass (inverted-pendulum walking).
//   * Vertical bob at 2x step frequency (each step pushes the body up).
//   * Legs are short and stiff - the swing leg lifts only ~0.04 body-heights
//     and pitches forward ~20 degrees.  Griffin & Kram explicitly note that
//     the high cost of penguin walking comes from the short legs, not the
//     waddle, so the leg swing amplitude is modest by design.
//   * Wings swing opposite to the legs, providing counter-rotation - this is
//     visible in nearly every recording of emperor penguins walking.
//   * The head does a gentle counter-sway of ~4 degrees to keep the eye
//     stable, a documented effect in bird locomotion (Necker, J. Comp.
//     Physiol. A, 2007, "Head-bobbing of walking birds").

#ifndef SKELETON_H
#define SKELETON_H

#include "math_utils.h"
#include "obj_loader.h"

#include <array>
#include <vector>
#include <cmath>
#include <string>

// Joint identifiers: fixed so we can build the walk driver by index.
//
// Semantic correction from the earlier draft of this file: after inspecting
// the bounding boxes of each of the 12 cuboids in the .obj (Y ranges, X ranges,
// Z ranges), part index 2 is the NECK (a thin collar between body and head),
// not the lower beak.  Parts 4 and 5 are the two beak segments.  The enum
// names below reflect this corrected semantics.
enum JointId : int {
    J_ROOT = 0,   // global translation + yaw; root of the tree
    J_PELVIS,     // waddle roll, forward lean (pitch); drives the body mesh
    J_BELLY,      // rigid child of the pelvis; drives the white belly panel
    J_NECK,       // between pelvis and head
    J_HEAD,       // head pitch + counter-sway yaw
    J_BEAK_A,     // upper beak segment
    J_BEAK_B,     // lower beak segment (the "jaw" - opens slightly as a cue)
    J_EYE_L,
    J_EYE_R,
    J_WING_L,     // shoulder pitch (swings fore/aft)
    J_WING_R,
    J_HIP_L,      // left-leg swing (pitch) + lift
    J_HIP_R,
    J_COUNT
};

struct Joint {
    std::string name;
    int         parent;          // -1 for the root
    Vec3        offset;          // translation from parent's origin (pose)
    Vec3        euler;           // current Euler angles (rad), Z * Y * X
    int         meshIndex{-1};   // -1 if this joint drives no mesh directly
};

struct Skeleton {
    std::array<Joint, J_COUNT> joints;
    std::array<Mat4,  J_COUNT> world; // per-frame world transforms

    // Build the tree.  Offsets are derived from each mesh's pivot so that
    // bind-pose rendering (all angles = 0) exactly reproduces the input model.
    //
    // General rule for the offset of a joint J with parent P:
    //     J.offset = meshes[J.meshIndex].pivot - meshes[P.meshIndex].pivot
    // because the rest-pose world position of J's origin is the mesh pivot,
    // and parent P's origin has already been placed at its own mesh pivot.
    // This keeps the rig invariant to any translation of the .obj in world
    // space at load time.
    void build(const std::vector<Mesh>& meshes) {
        auto& J = joints;

        // Bind-pose pivots (world-frame positions of each mesh's joint centre)
        const Vec3 bodyC    = meshes[0].pivot;
        const Vec3 bellyC   = meshes[1].pivot;
        const Vec3 neckC    = meshes[2].pivot;
        const Vec3 headC    = meshes[3].pivot;
        const Vec3 beakAC   = meshes[4].pivot;
        const Vec3 beakBC   = meshes[5].pivot;
        const Vec3 eyeLC    = meshes[6].pivot;
        const Vec3 eyeRC    = meshes[7].pivot;
        const Vec3 wingLC   = meshes[8].pivot;
        const Vec3 wingRC   = meshes[9].pivot;
        const Vec3 footLC   = meshes[10].pivot;
        const Vec3 footRC   = meshes[11].pivot;

        // -- ROOT: free in world.  Offset 0; animated translation added later.
        J[J_ROOT]   = {"root",   -1,       {0, 0, 0}, {0, 0, 0}, -1};

        // -- PELVIS: at body centre.  Waddle roll + pitch lean here.  Drives
        //            the main body (torso) mesh.
        J[J_PELVIS] = {"pelvis", J_ROOT,   bodyC,     {0, 0, 0}, 0};

        // -- BELLY: rigid child of the pelvis.  Offset = bellyC - bodyC so that
        //           in bind pose the belly mesh lands at its authored position.
        J[J_BELLY]  = {"belly",  J_PELVIS, bellyC - bodyC, {0,0,0}, 1};

        // -- NECK: between pelvis and head; rotates with the head chain.
        J[J_NECK]   = {"neck",   J_PELVIS, neckC  - bodyC, {0,0,0}, 2};

        // -- HEAD and attached parts: offsets in the parent's local frame.
        J[J_HEAD]   = {"head",   J_NECK,   headC  - neckC, {0,0,0}, 3};
        J[J_BEAK_A] = {"beakA",  J_HEAD,   beakAC - headC, {0,0,0}, 4};
        J[J_BEAK_B] = {"beakB",  J_HEAD,   beakBC - headC, {0,0,0}, 5};
        J[J_EYE_L]  = {"eye_L",  J_HEAD,   eyeLC  - headC, {0,0,0}, 6};
        J[J_EYE_R]  = {"eye_R",  J_HEAD,   eyeRC  - headC, {0,0,0}, 7};

        // -- WINGS: shoulder attachment (inner-top corner of the flipper).
        J[J_WING_L] = {"wing_L", J_PELVIS, wingLC - bodyC, {0,0,0}, 8};
        J[J_WING_R] = {"wing_R", J_PELVIS, wingRC - bodyC, {0,0,0}, 9};

        // -- FEET: hip attachment (top-rear corner of the foot cuboid).
        J[J_HIP_L]  = {"hip_L",  J_PELVIS, footLC - bodyC, {0,0,0}, 10};
        J[J_HIP_R]  = {"hip_R",  J_PELVIS, footRC - bodyC, {0,0,0}, 11};
    }

    // Assemble and cache the world transform for every joint.
    void updateWorld() {
        for (int i = 0; i < J_COUNT; ++i) {
            const Joint& j = joints[i];
            // Local transform: T(offset) * Rz * Ry * Rx
            Mat4 Rz = Mat4::rotationZ(j.euler.z);
            Mat4 Ry = Mat4::rotationY(j.euler.y);
            Mat4 Rx = Mat4::rotationX(j.euler.x);
            Mat4 local = Mat4::translation(j.offset) * Rz * Ry * Rx;
            world[i] = (j.parent < 0) ? local : world[j.parent] * local;
        }
    }
};

// ---------------------------------------------------------------------------
// Walk-cycle driver
// ---------------------------------------------------------------------------
//
// `period` = seconds per full cycle (two steps).  At t=0 the LEFT foot is
// about to plant, the right foot is mid-swing.  Forward direction = +Z.
struct WalkParams {
    float period          = 1.10f;   // seconds per 2-step cycle; penguins
                                     // typically walk at ~0.5 m/s with short
                                     // strides (Griffin & Kram, Nature 2000).
    float forwardSpeed    = 0.55f;   // world units per second along +Z
    float strideAmplitude = 0.35f;   // rad - hip fore/aft swing (~20 deg)
    float footLift        = 0.07f;   // vertical lift of swing foot (model units)
    float waddleRoll      = 0.22f;   // rad - ~12 deg, matches Griffin&Kram data
    float leanPitch       = 0.09f;   // rad - slight forward lean (~5 deg)
    float bobAmplitude    = 0.025f;  // vertical bob of the whole body
    float wingSwing       = 0.45f;   // rad - wing counter-swing amplitude
    float wingBase        = 0.20f;   // rad - wings held slightly out, always
    float headPitchAmp    = 0.05f;   // rad - subtle head bob
    float headYawAmp      = 0.04f;   // rad - counter-sway stabilising eye
    float beakOpenAmp     = 0.10f;   // rad - beak opens/closes for personality
    float beakOpenFreq    = 0.8f;    // Hz - slower than the step
};

// Apply the walk pose for time `t` to `sk`.  `elapsed` is used for the
// monotonic forward translation; `t` (= elapsed) is also the phase source.
inline void applyWalkPose(Skeleton& sk, float t, const WalkParams& wp)
{
    const float TWO_PI = 6.28318530717958647692f;
    float phi  = TWO_PI * (t / wp.period);    // full step-pair cycle
    float phi2 = 2.f * phi;                   // double-frequency (for bob)

    // ---- ROOT: translation + heading (straight line => yaw = 0) ----------
    sk.joints[J_ROOT].offset = { 0.f,
                                 wp.bobAmplitude * std::sin(phi2),
                                 wp.forwardSpeed * t };
    sk.joints[J_ROOT].euler  = { 0.f, 0.f, 0.f };

    // ---- PELVIS: waddle roll + forward lean ------------------------------
    // Roll (around forward Z): +ve when right foot is planted.  Classic
    // inverted-pendulum lateral rocking.
    float roll = wp.waddleRoll * std::sin(phi);
    // Gentle lean pitch (body tilts forward slightly when mid-stride).
    float lean = wp.leanPitch + 0.02f * std::sin(phi2);
    sk.joints[J_PELVIS].euler = { lean, 0.f, roll };

    // ---- HIPS: alternating fore/aft swing + foot-clear lift --------------
    // Left hip leads; right hip is phase-shifted by pi (half a cycle).
    float leftSwing  = wp.strideAmplitude * std::sin(phi);
    float rightSwing = wp.strideAmplitude * std::sin(phi + 3.14159265f);

    // Foot lift: only when the leg is in its forward-swing half, using the
    // squared positive-half sine so the lift is strictly non-negative and
    // peaks at mid-swing.  This is a standard ground-contact heuristic -
    // see Multon et al., "Computer Animation of Human Walking: A Survey",
    // Journal of Visualization and Computer Animation, 1999, §4.2.
    auto posSinSq = [](float a){ float s = std::sin(a); return s > 0.f ? s*s : 0.f; };
    float leftLift  = wp.footLift * posSinSq(phi);
    float rightLift = wp.footLift * posSinSq(phi + 3.14159265f);

    // Base offsets are preserved; we add a vertical lift to the hip.
    sk.joints[J_HIP_L].euler  = { leftSwing,  0.f, 0.f };
    sk.joints[J_HIP_R].euler  = { rightSwing, 0.f, 0.f };
    // Additive lift: the meshes live in a local frame around each hip, so
    // nudging the hip's own offset.y upwards raises the whole leg.
    sk.joints[J_HIP_L].offset.y += leftLift;
    sk.joints[J_HIP_R].offset.y += rightLift;
    // (Note: build() set the nominal offsets on construction; we re-apply
    //  those in the caller before each frame - see Animator::pose below.)

    // ---- WINGS: counter-swing (opposite phase to legs) -------------------
    float wingL = -wp.wingBase - wp.wingSwing * std::sin(phi);
    float wingR = -wp.wingBase - wp.wingSwing * std::sin(phi + 3.14159265f);
    // Hold-out: a small Z roll so the wings sit away from the torso and
    // clear the peak waddle roll (|waddleRoll| ~ 0.22 rad).  Without this,
    // the wings clip into the body at mid-stride.
    sk.joints[J_WING_L].euler = { wingL, 0.f, -0.28f };
    sk.joints[J_WING_R].euler = { wingR, 0.f,  0.28f };

    // ---- NECK + HEAD: counter-sway to stabilise gaze ---------------------
    // Birds are famous for keeping their head visually stable during
    // locomotion (Necker, J. Comp. Physiol. A, 2007 - "head-bobbing").
    // Model: most of the waddle-induced roll/yaw is compensated at the neck,
    // with a small residual head pitch at the step frequency.
    float headYaw   =  wp.headYawAmp   * std::sin(phi + 3.14159265f * 0.5f);
    float headPitch =  wp.headPitchAmp * std::sin(phi2);
    sk.joints[J_NECK].euler = { 0.f, 0.f, -roll * 0.30f };
    sk.joints[J_HEAD].euler = { headPitch, headYaw, 0.f };

    // ---- BEAK: subtle open/close; a touch of personality -----------------
    float beakPhase = TWO_PI * wp.beakOpenFreq * t;
    float beakOpen  = 0.5f * wp.beakOpenAmp * (1.f - std::cos(beakPhase));
    // The lower beak segment (part 5 in the .obj - the one below the ridge)
    // rotates open about its rear edge.
    sk.joints[J_BEAK_B].euler = { -beakOpen, 0.f, 0.f };
    sk.joints[J_BEAK_A].euler = { 0.f, 0.f, 0.f };

    // ---- Eyes / belly: inherit rigidly from parents ----------------------
    sk.joints[J_BELLY].euler = {0, 0, 0};
    sk.joints[J_EYE_L].euler = {0, 0, 0};
    sk.joints[J_EYE_R].euler = {0, 0, 0};
}

// ---------------------------------------------------------------------------
// Animator: wraps a skeleton and re-applies bind offsets on each pose update,
// so the additive per-frame changes in applyWalkPose do not accumulate.
// ---------------------------------------------------------------------------
class Animator {
public:
    void init(const std::vector<Mesh>& meshes) {
        sk.build(meshes);
        // Record the bind offsets so we can restore them each frame.
        for (int i = 0; i < J_COUNT; ++i) bindOffsets[i] = sk.joints[i].offset;
    }

    // Pose at simulation time t (seconds).  After this call sk.world is ready
    // to be used by the rasteriser.
    void pose(float t, const WalkParams& wp) {
        for (int i = 0; i < J_COUNT; ++i) sk.joints[i].offset = bindOffsets[i];
        applyWalkPose(sk, t, wp);
        sk.updateWorld();
    }

    Skeleton sk;
private:
    std::array<Vec3, J_COUNT> bindOffsets;
};

#endif // SKELETON_H
