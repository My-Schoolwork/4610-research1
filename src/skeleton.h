// skeleton.h - Hierarchical articulated skeleton and the procedural walk-cycle
// driver for the Cubism penguin model.
//
// Skeleton model
// --------------
// Each Joint owns a local transform T_local(t) = T(offset) * R(euler(t)) where
// the offset is the pose-space translation from the parent's origin and
// R(euler) is built from per-axis Euler rotations (extrinsic Z * Y * X in our
// convention).  The world transform is the usual left-multiplicative chain
// T_world[i] = T_world[parent(i)] * T_local[i]
//
// Walk-cycle driver
// -----------------
// The gait is procedural: every DOF is a closed-form function of a phase
// angle phi = 2*pi * f * t, where f is the step frequency.  This follows the
// "cosine-family" controller used extensively in real-time games and
// biomechanics models 

#ifndef SKELETON_H
#define SKELETON_H

#include "math_utils.h"
#include "obj_loader.h"

#include <array>
#include <vector>
#include <cmath>
#include <string>

// Joint identifiers: fixed so we can build the walk driver by index.
// Semantic correction from the earlier draft of this file: after inspecting
// the bounding boxes of each of the 12 cuboids in the .obj (Y ranges, X ranges,
// Z ranges), part index 2 is the NECK (a thin collar between body and head),
// not the lower beak.  Parts 4 and 5 are the two beak segments.  The enum
// names below reflect this corrected semantics.
enum JointId : int {
    J_ROOT = 0, // global translation + yaw; root of the tree
    J_PELVIS, // waddle roll, forward lean (pitch); drives the body mesh
    J_BELLY, // rigid child of the pelvis; drives the white belly panel
    J_NECK, // between pelvis and head
    J_HEAD, // head pitch + counter-sway yaw
    J_BEAK_A, // upper beak segment
    J_BEAK_B, // lower beak segment (the "jaw" - opens slightly as a cue)
    J_EYE_L, // left eye
    J_EYE_R, // right eye
    J_WING_L, // shoulder pitch (swings fore/aft)
    J_WING_R, // shoulder pitch (swings fore/aft)
    J_HIP_L, // left-leg swing (pitch) + lift
    J_HIP_R, // right-leg swing (pitch) + lift
    J_COUNT 
};

struct Joint {
    std::string name;
    int parent; // -1 for the root
    Vec3 offset; // translation from parent's origin (pose)
    Vec3 euler;  // current Euler angles (rad), Z * Y * X
    int meshIndex{-1}; // -1 if this joint drives no mesh directly
};

struct Skeleton {
    std::array<Joint, J_COUNT> joints;
    std::array<Mat4,  J_COUNT> world; // per-frame world transforms

    // Build the tree.  Offsets are derived from each mesh's pivot so that
    // bind-pose rendering (all angles = 0) exactly reproduces the input model.
    // General rule for the offset of a joint J with parent P: J.offset = meshes[J.meshIndex].pivot - meshes[P.meshIndex].pivot
    // because the rest-pose world position of J's origin is the mesh pivot,
    // and parent P's origin has already been placed at its own mesh pivot.
    // This keeps the rig invariant to any translation of the .obj in world
    // space at load time.
    void build(const std::vector<Mesh>& meshes) {
        auto& J = joints;
        // Mesh indices for penguin_fixed.obj (from penguinPartSpecs ordering):
        //   0 = foot_L   1 = body      2 = belly     3 = neck
        //   4 = head      5 = beak_upper  6 = beak_lower
        //   7 = eye_L     8 = eye_R     9 = wing_L   10 = wing_R   11 = foot_R
        const Vec3 footLC   = meshes[0].pivot; // foot_L: top-rear hip pivot
        const Vec3 bodyC    = meshes[1].pivot; // body:   centre of torso
        const Vec3 bellyC   = meshes[2].pivot; // belly:  centre of front panel
        const Vec3 neckC    = meshes[3].pivot; // neck:   collar centre
        const Vec3 headC    = meshes[4].pivot; // head:   centre of head cube
        const Vec3 beakAC   = meshes[5].pivot; // beak_upper
        const Vec3 beakBC   = meshes[6].pivot; // beak_lower
        const Vec3 eyeLC    = meshes[7].pivot; // eye_L
        const Vec3 eyeRC    = meshes[8].pivot; // eye_R
        const Vec3 wingLC   = meshes[9].pivot; // wing_L: inner-top shoulder
        const Vec3 wingRC   = meshes[10].pivot; // wing_R: inner-top shoulder
        const Vec3 footRC   = meshes[11].pivot; // foot_R: top-rear hip pivot

        // ROOT: free in world.  Offset 0; animated translation added later.
        J[J_ROOT]   = {"root",   -1,       {0, 0, 0}, {0, 0, 0}, -1};

        // PELVIS: at body centre.  Waddle roll + pitch lean here.  Drives
        //            the main body (torso) mesh.
        J[J_PELVIS] = {"pelvis", J_ROOT,   bodyC,          {0, 0, 0}, 1};

        // BELLY: rigid child of the pelvis.
        J[J_BELLY]  = {"belly",  J_PELVIS, bellyC - bodyC, {0,0,0},   2};

        // NECK: between pelvis and head; counter-sways with walking roll.
        J[J_NECK]   = {"neck",   J_PELVIS, neckC  - bodyC, {0,0,0},   3};

        // HEAD and attached parts: offsets in the parent's local frame.
        J[J_HEAD]   = {"head",   J_NECK,   headC  - neckC, {0,0,0},   4};
        J[J_BEAK_A] = {"beakA",  J_HEAD,   beakAC - headC, {0,0,0},   5};
        J[J_BEAK_B] = {"beakB",  J_HEAD,   beakBC - headC, {0,0,0},   6};
        J[J_EYE_L]  = {"eye_L",  J_HEAD,   eyeLC  - headC, {0,0,0},   7};
        J[J_EYE_R]  = {"eye_R",  J_HEAD,   eyeRC  - headC, {0,0,0},   8};

        // WINGS: shoulder attachment (inner-top corner of the flipper).
        J[J_WING_L] = {"wing_L", J_PELVIS, wingLC - bodyC, {0,0,0},   9};
        J[J_WING_R] = {"wing_R", J_PELVIS, wingRC - bodyC, {0,0,0},  10};

        // FEET: hip attachment (top-rear corner of the foot cuboid).
        J[J_HIP_L]  = {"hip_L",  J_PELVIS, footLC - bodyC, {0,0,0},   0};
        J[J_HIP_R]  = {"hip_R",  J_PELVIS, footRC - bodyC, {0,0,0},  11};
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

// Walk-cycle driver
// `period` = seconds per full cycle (two steps).  At t=0 the LEFT foot is
// about to plant, the right foot is mid-swing.  Forward direction = +Z.
struct WalkParams {
    float period = 1.10f; // seconds per 2-step cycle
    float forwardSpeed = 0.55f; // world units / second (used only in non-interactive mode)
    float strideAmplitude = 0.35f; // rad - maximum hip fore/aft swing
    float footLift = 0.07f; // vertical lift of swing foot (model units)
    float waddleRoll = 0.22f; // rad - ~12 deg lateral torso roll
    float leanPitch = 0.09f; // rad - forward lean at full speed
    float bobAmplitude = 0.022f; // vertical COM rise per step (only upward)
    float wingSwing = 0.40f; // rad - wing fore/aft swing amplitude
    float wingFlapAmp = 0.30f; // rad - wing up/down flap amplitude
    float wingBase = 0.22f; // rad - wings held out at rest
    float headPitchAmp = 0.04f; // rad - head nod per gait cycle
    float headYawAmp = 0.04f; // rad - head counter-yaw
    float beakOpenAmp = 0.25f; // rad - beak open/close range
    float beakOpenFreq = 0.8f;  // Hz  - beak idle frequency

    // Set by the caller each frame; drive speed-dependent and idle animations.
    float speedRatio = 0.f; // 0 = standing, 1 = full speed
    float idleTime = 0.f; // always-advancing real time (for idle anims)
};

// Apply the walk pose for time `t` to `sk`.
// `t` is the animation phase (advances proportionally to walk speed).
// `wp.idleTime` is always-advancing real time used for idle animations.
// `wp.speedRatio` is 0 (idle) to 1 (full speed) for blending.
inline void applyWalkPose(Skeleton& sk, float t, const WalkParams& wp)
{
    const float PI = 3.14159265f;
    const float TWO_PI = 2.f * PI;

    float phi = TWO_PI * (t / wp.period); // gait cycle phase (0..2π per 2 steps)
    float phi2 = 2.f * phi; // double-frequency (once per step)
    float sr = wp.speedRatio; // shorthand

    // Blend factor for idle animations: full at rest, fades out as speed rises.
    float idle = std::fmax(0.f, 1.f - sr * 4.f);

    // ROOT: vertical bob (positive only so feet never sink below ground)
    // Using |sin(phi2)| gives two gentle lifts per gait cycle without the
    // downward dip that would push feet through the ice plane.
    float walkBob = wp.bobAmplitude * std::fabs(std::sin(phi2)) * sr;
    float idleBob = 0.007f * std::sin(wp.idleTime * 1.4f) * idle;
    sk.joints[J_ROOT].offset = { 0.f, walkBob + idleBob, wp.forwardSpeed * t };
    sk.joints[J_ROOT].euler  = { 0.f, 0.f, 0.f };

    // PELVIS: waddle roll + forward lean
    // Roll only while walking; idle sway substitutes when standing.
    float roll = wp.waddleRoll * std::sin(phi) * sr;
    float lean = wp.leanPitch  * sr + 0.02f * std::sin(phi2) * sr;
    float idleSway = 0.035f * std::sin(wp.idleTime * 0.70f) * idle;
    sk.joints[J_PELVIS].euler = { lean, 0.f, roll + idleSway };

    // HIPS: alternating fore/aft swing + foot-clear lift
    // Stride amplitude grows with speed (short careful steps when slow, wide
    // strides at full pace) – matches speed-amplitude coupling in Cavagna et al.
    float stride = wp.strideAmplitude * (0.45f + 0.55f * sr);
    float lift = wp.footLift * (0.40f + 0.60f * sr);

    float leftSwing  = stride * std::sin(phi);
    float rightSwing = stride * std::sin(phi + PI);

    // Lift is strictly non-negative (squared positive half-sine).
    auto posSinSq = [](float a){ float s = std::sin(a); return s > 0.f ? s*s : 0.f; };
    float leftLift  = lift * posSinSq(phi);
    float rightLift = lift * posSinSq(phi + PI);

    sk.joints[J_HIP_L].euler    = { leftSwing,  0.f, 0.f };
    sk.joints[J_HIP_R].euler    = { rightSwing, 0.f, 0.f };
    sk.joints[J_HIP_L].offset.y += leftLift;
    sk.joints[J_HIP_R].offset.y += rightLift;

    // WINGS: counter-swing opposite to the legs 
    // At rest, wings are held further out (wingBase grows when idle).
    // Swing scales with speed; a subtle Z-tilt varies with the pitch swing,
    // giving a natural "tip forward as the wing reaches back" motion.
    float wingBase = wp.wingBase + 0.06f * idle;
    float swing = wp.wingSwing * sr;

    float wingL = -(wingBase + swing * std::sin(phi));
    float wingR = -(wingBase + swing * std::sin(phi + PI));
    // Z tilt: wing tips slightly forward/back as the flipper swings
    // Base hold-out is 0.42 so that at maximum flap-down the wing still
    // has ~0.12 rad of clearance and doesn't clip into the torso.
    float wingFlap = wp.wingFlapAmp * std::sin(phi2);
    float rollL = -0.42f - 0.06f * std::cos(phi) - wingFlap;
    float rollR =  0.42f + 0.06f * std::cos(phi + PI) + wingFlap;
    sk.joints[J_WING_L].euler = { wingL, 0.f, rollL };
    sk.joints[J_WING_R].euler = { wingR, 0.f, rollR };

    // NECK + HEAD: gaze stabilisation
    // Neck absorbs most of the waddle-induced roll (30 %).
    // Head pitch uses gait-cycle frequency (once per stride, not per step)
    // for a slow, naturalistic nod.  Idle: gentle curiosity tilt.
    float headYaw   = wp.headYawAmp   * std::sin(phi + PI * 0.5f) * sr;
    float headPitch = wp.headPitchAmp * std::sin(phi) * sr;
    float idleHead  = 0.030f * std::sin(wp.idleTime * 0.50f) * idle;
    sk.joints[J_NECK].euler = { 0.f, 0.f, -roll * 0.30f };
    sk.joints[J_HEAD].euler = { headPitch, headYaw + idleHead, 0.f };

    // BEAK: idle chatter (driven by real time so it plays even at speed=0)
    float beakPhase = TWO_PI * wp.beakOpenFreq * wp.idleTime;
    float beakOpen  = 0.5f * wp.beakOpenAmp * (1.f - std::cos(beakPhase));
    sk.joints[J_BEAK_B].euler = { -beakOpen,        0.f, 0.f };  // lower jaw drops
    sk.joints[J_BEAK_A].euler = {  beakOpen * 0.4f, 0.f, 0.f };  // upper beak lifts

    // Eyes / belly: inherit rigidly from parents
    sk.joints[J_BELLY].euler = {0, 0, 0};
    sk.joints[J_EYE_L].euler = {0, 0, 0};
    sk.joints[J_EYE_R].euler = {0, 0, 0};
}

// Animator: wraps a skeleton and re-applies bind offsets on each pose update,
// so the additive per-frame changes in applyWalkPose do not accumulate.
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

// Backflip animation state machine
// Phases:
// 0 - Idle (not doing a backflip)
// 1 - Crouch (prepare for jump)
// 2 - Launch (push off ground, start rising)
// 3 - Airborne (body rotates backward, parabolic arc)
// 4 - Descend (falling, completing rotation)
// 5 - Land (impact crouch)
// 6 - Recover (stand back up)
// The backflip is a full backward rotation around the local X axis (pitch),
// combined with a vertical parabolic arc for the root translation.
// Smooth easing uses cosine interpolation for natural motion.

enum BackflipPhase : int {
    BF_IDLE = 0,
    BF_CROUCH,
    BF_LAUNCH,
    BF_AIRBORNE,
    BF_DESCEND,
    BF_LAND,
    BF_RECOVER
};

struct BackflipState {
    BackflipPhase phase = BF_IDLE;
    float phaseTime = 0.f; // time elapsed in current phase

    // Phase durations (seconds)
    static constexpr float CROUCH_DUR = 0.25f;
    static constexpr float LAUNCH_DUR = 0.15f;
    static constexpr float AIRBORNE_DUR = 0.40f;
    static constexpr float DESCEND_DUR = 0.20f;
    static constexpr float LAND_DUR = 0.15f;
    static constexpr float RECOVER_DUR = 0.30f;

    // Jump parameters
    static constexpr float JUMP_HEIGHT = 1.8f; // peak height in world units
    static constexpr float CROUCH_DEPTH = -0.15f; // how much the penguin squats
    static constexpr float LAND_DEPTH = -0.12f; // landing impact crouch

    bool isActive() const { return phase != BF_IDLE; }

    void trigger() {
        if (phase == BF_IDLE) {
            phase = BF_CROUCH;
            phaseTime = 0.f;
        }
    }

    void update(float dt) {
        if (phase == BF_IDLE) return;
        phaseTime += dt;

        // Advance through phases based on timing
        switch (phase) {
        case BF_CROUCH:
            if (phaseTime >= CROUCH_DUR) { phase = BF_LAUNCH; phaseTime -= CROUCH_DUR; }
            break;
        case BF_LAUNCH:
            if (phaseTime >= LAUNCH_DUR) { phase = BF_AIRBORNE; phaseTime -= LAUNCH_DUR; }
            break;
        case BF_AIRBORNE:
            if (phaseTime >= AIRBORNE_DUR) { phase = BF_DESCEND; phaseTime -= AIRBORNE_DUR; }
            break;
        case BF_DESCEND:
            if (phaseTime >= DESCEND_DUR) { phase = BF_LAND; phaseTime -= DESCEND_DUR; }
            break;
        case BF_LAND:
            if (phaseTime >= LAND_DUR) { phase = BF_RECOVER; phaseTime -= LAND_DUR; }
            break;
        case BF_RECOVER:
            if (phaseTime >= RECOVER_DUR) { phase = BF_IDLE; phaseTime = 0.f; }
            break;
        default: break;
        }
    }

    // Smooth ease-in-out using cosine interpolation
    static float easeInOut(float t) {
        return 0.5f * (1.f - std::cos(t * 3.14159265f));
    }
    // Ease out (decelerating)
    static float easeOut(float t) {
        return std::sin(t * 3.14159265f * 0.5f);
    }
    // Ease in (accelerating)
    static float easeIn(float t) {
        return 1.f - std::cos(t * 3.14159265f * 0.5f);
    }

    // Get the vertical offset for the root (Y translation)
    float getHeightOffset() const {
        switch (phase) {
        case BF_CROUCH: {
            float t = easeInOut(phaseTime / CROUCH_DUR);
            return CROUCH_DEPTH * t;
        }
        case BF_LAUNCH: {
            // From crouch depth up to ~30% of jump height
            float t = easeIn(phaseTime / LAUNCH_DUR);
            return CROUCH_DEPTH * (1.f - t) + JUMP_HEIGHT * 0.3f * t;
        }
        case BF_AIRBORNE: {
            // Parabolic arc from 30% to peak (100%) back to ~70%
            float t = phaseTime / AIRBORNE_DUR;
            // Quadratic: peaks at t=0.5
            float arc = -4.f * (t - 0.5f) * (t - 0.5f) + 1.f;
            float base = 0.3f + 0.7f * arc; // ranges from 0.3 to 1.0 back to 0.3
            return JUMP_HEIGHT * base;
        }
        case BF_DESCEND: {
            // From ~30% of height down to 0
            float t = easeIn(phaseTime / DESCEND_DUR);
            return JUMP_HEIGHT * 0.3f * (1.f - t) + LAND_DEPTH * t;
        }
        case BF_LAND: {
            // Impact crouch then start recovering
            float t = easeOut(phaseTime / LAND_DUR);
            return LAND_DEPTH * (1.f - t);
        }
        case BF_RECOVER: {
            // Already at 0 basically, just smooth out any residual
            float t = easeOut(phaseTime / RECOVER_DUR);
            (void)t;
            return 0.f;
        }
        default: return 0.f;
        }
    }

    // Get the backward rotation angle (radians, around X axis / pitch)
    // Full rotation = -2*PI (backward flip)
    float getRotation() const {
        const float PI = 3.14159265f;
        const float FULL_ROTATION = -2.f * PI;  // negative = backward

        switch (phase) {
        case BF_CROUCH: {
            // Slight forward lean as preparation
            float t = easeInOut(phaseTime / CROUCH_DUR);
            return 0.15f * t;  // lean forward slightly
        }
        case BF_LAUNCH: {
            // Snap back from forward lean, start backward rotation
            float t = easeIn(phaseTime / LAUNCH_DUR);
            return 0.15f * (1.f - t) + FULL_ROTATION * 0.05f * t;
        }
        case BF_AIRBORNE: {
            // Main rotation: 5% to 80% of full rotation
            float t = easeInOut(phaseTime / AIRBORNE_DUR);
            return FULL_ROTATION * (0.05f + 0.75f * t);
        }
        case BF_DESCEND: {
            // Complete rotation: 80% to 100%
            float t = easeOut(phaseTime / DESCEND_DUR);
            return FULL_ROTATION * (0.80f + 0.20f * t);
        }
        case BF_LAND: {
            // At full rotation (= 0 mod 2pi), slight overshoot
            return FULL_ROTATION;
        }
        case BF_RECOVER: {
            // Rotation is complete, no residual
            return FULL_ROTATION;
        }
        default: return 0.f;
        }
    }

    // Get pelvis crouch angle (extra forward lean during crouch/land)
    float getPelvisLean() const {
        switch (phase) {
        case BF_CROUCH: {
            float t = easeInOut(phaseTime / CROUCH_DUR);
            return 0.3f * t;  // crouch lean
        }
        case BF_LAUNCH: {
            float t = easeIn(phaseTime / LAUNCH_DUR);
            return 0.3f * (1.f - t);  // un-crouch
        }
        case BF_LAND: {
            float t = phaseTime / LAND_DUR;
            // Impact lean: sharp then fade
            return 0.25f * (1.f - t);
        }
        case BF_RECOVER: {
            float t = easeOut(phaseTime / RECOVER_DUR);
            return 0.05f * (1.f - t);
        }
        default: return 0.f;
        }
    }

    // Get wing spread (wings go out during the flip)
    float getWingSpread() const {
        switch (phase) {
        case BF_CROUCH: {
            float t = easeInOut(phaseTime / CROUCH_DUR);
            return 0.3f * t;
        }
        case BF_LAUNCH:
        case BF_AIRBORNE:
        case BF_DESCEND:
            return 0.3f;  // wings fully spread in air
        case BF_LAND: {
            float t = easeOut(phaseTime / LAND_DUR);
            return 0.3f * (1.f - t);
        }
        case BF_RECOVER: {
            float t = easeOut(phaseTime / RECOVER_DUR);
            return 0.05f * (1.f - t);
        }
        default: return 0.f;
        }
    }

    // Get leg tuck (legs pull up during flip)
    float getLegTuck() const {
        switch (phase) {
        case BF_CROUCH: {
            float t = easeInOut(phaseTime / CROUCH_DUR);
            return -0.4f * t;  // bend knees
        }
        case BF_LAUNCH: {
            float t = easeIn(phaseTime / LAUNCH_DUR);
            return -0.4f + (-0.3f) * t;  // tuck tighter
        }
        case BF_AIRBORNE:
            return -0.7f;  // fully tucked
        case BF_DESCEND: {
            // Extend legs for landing
            float t = easeOut(phaseTime / DESCEND_DUR);
            return -0.7f * (1.f - t);
        }
        case BF_LAND: {
            // Absorb impact
            float t = phaseTime / LAND_DUR;
            return -0.3f * (1.f - t);
        }
        case BF_RECOVER: {
            float t = easeOut(phaseTime / RECOVER_DUR);
            return -0.05f * (1.f - t);
        }
        default: return 0.f;
        }
    }
};

// Slide animation state machine
// Phases:
//   SL_IDLE - not sliding
//   SL_ENTER - dive: penguin hops up then belly-flops (~0.50 s) hopY arcs +HOP_HEIGHT during first 45% of enter (airborne), body pitch (blend) only activates in the landing 55%
//   SL_ACTIVE - gliding on ice; stays until user presses G again
//   SL_EXIT - transition: lying -> standing (~0.40 s)

enum SlidePhase : int { SL_IDLE = 0, SL_ENTER, SL_ACTIVE, SL_EXIT };

struct SlideState {
    SlidePhase phase = SL_IDLE;
    float phaseTime = 0.f;
    float blend = 0.f;   // 0 = standing, 1 = fully lying
    float hopY = 0.f;   // live upward offset during the dive arc
    float slideSpeed = 0.f; // current forward velocity (world units/s)
    float blendAtExit = 1.f; // blend captured when exit begins (for cancel)

    static constexpr float ENTER_DUR = 0.50f;  // longer to fit hop arc
    static constexpr float EXIT_DUR = 0.40f;
    static constexpr float HOP_HEIGHT = 0.30f;  // peak height above ground during dive
    static constexpr float LAUNCH_BONUS = 2.0f;   // speed added on top of current velocity
    static constexpr float FRICTION = 1.4f;   // deceleration (units/s²)

    // Pose targets (full blend = 1)
    static constexpr float BODY_PITCH = 1.45f;  // rad – body almost horizontal (~83°)
    static constexpr float HEAD_PITCH = -0.75f; // rad – head tilts back to look up
    static constexpr float ROOT_Y = -0.34f; // root drops so belly skims ice
    static constexpr float WING_Z_ADD =  0.40f; // extra Z spread (wings out wide)
    static constexpr float LEG_PITCH = -0.35f; // legs swept back

    bool isActive()  const { return phase != SL_IDLE; }
    bool isSliding() const { return phase == SL_ACTIVE; }

    // Toggle: starts dive-slide from idle, or cancels from any active phase.
    void trigger(float currentSpeed) {
        if (phase == SL_IDLE) {
            phase      = SL_ENTER;
            phaseTime  = 0.f;
            blend      = 0.f;
            hopY       = 0.f;
            // Carry full current speed plus a launch kick
            slideSpeed = std::fabs(currentSpeed) + LAUNCH_BONUS;
        } else {
            blendAtExit = blend;
            hopY        = 0.f;  // abort any mid-air arc immediately
            phase       = SL_EXIT;
            phaseTime   = 0.f;
        }
    }

    void update(float dt) {
        if (phase == SL_IDLE) return;
        phaseTime += dt;
        switch (phase) {
        case SL_ENTER: {
            float t = std::fmin(phaseTime / ENTER_DUR, 1.f);
            // Hop arc: a sine bump that peaks at t=0.5, gone by t=1
            hopY = std::sin(t * 3.14159265f) * HOP_HEIGHT;
            // Body pitch (blend) only starts at the landing phase (t > 0.45)
            float blendT = std::fmax(0.f, (t - 0.45f) / 0.55f);
            blend = easeInOut(blendT);
            if (phaseTime >= ENTER_DUR) {
                blend = 1.f; hopY = 0.f;
                phase = SL_ACTIVE; phaseTime = 0.f;
            }
            break;
        }
        case SL_ACTIVE:
            blend      = 1.f;
            slideSpeed = std::fmax(0.f, slideSpeed - FRICTION * dt);
            break;
        case SL_EXIT:
            blend = blendAtExit * (1.f - easeInOut(std::fmin(phaseTime / EXIT_DUR, 1.f)));
            if (phaseTime >= EXIT_DUR) { blend = 0.f; phase = SL_IDLE; phaseTime = 0.f; slideSpeed = 0.f; }
            break;
        default: break;
        }
    }

    static float easeInOut(float t) { return 0.5f * (1.f - std::cos(t * 3.14159265f)); }
};

// Apply slide pose overrides AFTER applyWalkPose.
inline void applySlidePose(Skeleton& sk, const SlideState& sl)
{
    if (!sl.isActive()) return;
    float b = sl.blend;

    // Root: drops to ice at full blend, but arcs upward during the dive hop.
    sk.joints[J_ROOT].offset.y += b * SlideState::ROOT_Y + sl.hopY;

    // Body pitches almost horizontal; fade out the walk waddle roll.
    sk.joints[J_PELVIS].euler.x  = b * SlideState::BODY_PITCH;
    sk.joints[J_PELVIS].euler.z *= (1.f - b);

    // Neck straightens (kill the counter-sway roll).
    sk.joints[J_NECK].euler.z *= (1.f - b);

    // Head tilts back relative to the now-horizontal body → looks upward.
    sk.joints[J_HEAD].euler.x  = b * SlideState::HEAD_PITCH;
    sk.joints[J_HEAD].euler.y *= (1.f - b);

    // Beak opens slightly – a happy tobogganing expression.
    float bk = 0.12f * b;
    sk.joints[J_BEAK_A].euler.x =  bk * 0.4f;
    sk.joints[J_BEAK_B].euler.x = -bk;

    // Wings: stop fore/aft swing, spread wide to sides.
    sk.joints[J_WING_L].euler.x *= (1.f - b);
    sk.joints[J_WING_R].euler.x *= (1.f - b);
    sk.joints[J_WING_L].euler.z -= b * SlideState::WING_Z_ADD;
    sk.joints[J_WING_R].euler.z += b * SlideState::WING_Z_ADD;

    // Legs sweep back.
    sk.joints[J_HIP_L].euler.x = b * SlideState::LEG_PITCH;
    sk.joints[J_HIP_R].euler.x = b * SlideState::LEG_PITCH;
}

// Apply backflip pose overrides to the skeleton.
// Call this AFTER applyWalkPose to override the relevant joints.
inline void applyBackflipPose(Skeleton& sk, const BackflipState& bf)
{
    if (!bf.isActive()) return;

    float heightOff = bf.getHeightOffset();
    float rotation  = bf.getRotation();
    float pelvisLean = bf.getPelvisLean();
    float wingSpread = bf.getWingSpread();
    float legTuck   = bf.getLegTuck();

    // Override root Y offset for jump arc
    sk.joints[J_ROOT].offset.y += heightOff;

    // Apply backward rotation to the pelvis (whole body rotates)
    sk.joints[J_PELVIS].euler.x += rotation + pelvisLean;

    // Wings spread outward during flip
    sk.joints[J_WING_L].euler.z -= wingSpread;
    sk.joints[J_WING_R].euler.z += wingSpread;

    // Tuck legs during airborne phases
    sk.joints[J_HIP_L].euler.x += legTuck;
    sk.joints[J_HIP_R].euler.x += legTuck;
}

#endif // SKELETON_H
