// obj_loader.h - Minimal Wavefront .obj loader specialised for the Cubism
// penguin model supplied with the assignment.
//
// The .obj format is documented in the original Wavefront Advanced Visualizer
// manual (Wavefront Technologies, 1992) - a useful modern reference is Paul
// Bourke's summary at http://paulbourke.net/dataformats/obj/.  We implement
// only the subset we need: "v" (vertex) and "f" (triangle face) lines.
//
// The penguin model is authored as 12 axis-aligned cuboids, 8 vertices each,
// listed consecutively.  We exploit this by segmenting the vertex list into
// body parts and emitting one Mesh per part.  Each Mesh also stores its own
// rest-pose "pivot" - the point about which joint rotations are applied by
// the skeleton (Section 2 of the accompanying report).

#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "math_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>

// A triangle stored as three indices into the mesh-local vertex list.
struct Tri { int a, b, c; };

struct Mesh {
    std::string         name;       // human-readable body-part name
    std::vector<Vec3>   vertices;   // in the mesh's REST-POSE local frame
    std::vector<Tri>    faces;      // triangle list (CCW winding)
    Vec3                pivot{};    // joint centre in the model's world frame
    std::array<float,3> colorRgb{}; // diffuse colour for flat shading
};

// The 12 cuboids, ordered as they appear in the file.  Names and colours are
// chosen to mimic an emperor penguin's counter-shaded livery (black back,
// white belly, yellow-orange beak, dark eyes) - see IUCN Red List entry for
// Aptenodytes forsteri, BirdLife International, 2020.
struct PartSpec {
    const char*          name;
    std::array<float,3>  color;
};

inline const std::vector<PartSpec>& penguinPartSpecs() {
    // Corrected part naming (verified against the .obj bounding boxes):
    //   part 2 is a thin collar between body and head - the NECK - not the
    //   beak.  Parts 4 and 5 are the two beak segments (upper ridge and main
    //   beak body respectively).  The colours below reflect this, giving the
    //   classic emperor-penguin livery: black back, white belly, orange beak
    //   and feet.  (IUCN Red List entry for Aptenodytes forsteri, BirdLife
    //   International, 2020.)
    static const std::vector<PartSpec> s = {
        {"body",        {0.12f, 0.12f, 0.14f}}, //  0: black main torso
        {"belly",       {0.97f, 0.97f, 0.97f}}, //  1: white chest/belly panel
        {"neck",        {0.12f, 0.12f, 0.14f}}, //  2: black neck collar
        {"head",        {0.12f, 0.12f, 0.14f}}, //  3: black head
        {"beak_upper",  {0.98f, 0.65f, 0.10f}}, //  4: orange upper beak ridge
        {"beak_lower",  {0.98f, 0.55f, 0.08f}}, //  5: orange lower beak (jaw)
        {"eye_L",       {0.05f, 0.05f, 0.05f}}, //  6: left eye
        {"eye_R",       {0.05f, 0.05f, 0.05f}}, //  7: right eye
        {"wing_L",      {0.10f, 0.10f, 0.12f}}, //  8: left flipper
        {"wing_R",      {0.10f, 0.10f, 0.12f}}, //  9: right flipper
        {"foot_L",      {0.95f, 0.55f, 0.10f}}, // 10: left foot
        {"foot_R",      {0.95f, 0.55f, 0.10f}}  // 11: right foot
    };
    return s;
}

// Load the penguin .obj and emit one Mesh per 8-vertex block.
//
// The rest-pose pivot for each part is placed at a point that makes its joint
// rotation behave naturally:
//   * wings  -> top-inside corner (shoulder attachment to the torso)
//   * feet   -> top-rear corner (hip attachment)
//   * other  -> bounding-box centre
// For the rotated parts we also translate the stored vertices into the local
// joint frame (v_local = v_world - pivot) so that multiplying by a rotation
// matrix pivots them correctly.  Non-rotated / small-rotation parts keep the
// bounding-box centre as pivot.
//
// Returns true on success.
inline bool loadPenguinObj(const std::string& path, std::vector<Mesh>& outMeshes)
{
    std::vector<Vec3> allVerts;
    std::vector<std::array<int,3>> allFaces; // 1-based indices as per .obj

    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3)
                allVerts.push_back({x, y, z});
        } else if (line[0] == 'f' && line[1] == ' ') {
            int a, b, c;
            if (std::sscanf(line + 2, "%d %d %d", &a, &b, &c) == 3)
                allFaces.push_back({a, b, c});
        }
    }
    std::fclose(f);

    if (allVerts.size() != 96) {
        std::fprintf(stderr, "Expected 96 vertices, got %zu\n", allVerts.size());
        return false;
    }

    const auto& specs = penguinPartSpecs();
    outMeshes.clear();
    outMeshes.reserve(specs.size());

    // Pivot placement per part (indices into 8-vertex block):
    //   vertex layout in each block follows the .obj - corners of a cuboid.
    // We look up the bounding box from the vertices themselves; the choice
    // of pivot is then expressed as a fractional position inside that box.
    struct PivotRule { float fx, fy, fz; }; // in [0, 1]^3 of the local AABB
    //  body, belly, beak_lower, head,     beak_upperA, beak_upperB,
    //  eye_L, eye_R, wing_L,    wing_R,   foot_L,     foot_R
    std::array<PivotRule, 12> rules = {{
        {0.5f, 0.5f, 0.5f},   //  0 body   - rotate around its centre
        {0.5f, 0.5f, 0.5f},   //  1 belly
        {0.5f, 0.5f, 0.0f},   //  2 lower beak
        {0.5f, 0.0f, 0.5f},   //  3 head   - pivot at base of neck
        {0.5f, 0.5f, 0.0f},   //  4 beak upper A
        {0.5f, 0.5f, 0.0f},   //  5 beak upper B
        {0.5f, 0.5f, 0.5f},   //  6 eye L
        {0.5f, 0.5f, 0.5f},   //  7 eye R
        {1.0f, 1.0f, 0.5f},   //  8 wing_L - shoulder: INNER top corner
        {0.0f, 1.0f, 0.5f},   //  9 wing_R - shoulder: INNER top corner
        {0.5f, 1.0f, 0.0f},   // 10 foot_L - hip: top-rear
        {0.5f, 1.0f, 0.0f},   // 11 foot_R - hip: top-rear
    }};

    for (size_t p = 0; p < specs.size(); ++p) {
        Mesh mesh;
        mesh.name     = specs[p].name;
        mesh.colorRgb = specs[p].color;

        // 8 vertices for this block
        std::vector<Vec3> block(allVerts.begin() + p * 8,
                                allVerts.begin() + (p + 1) * 8);

        // AABB
        Vec3 mn = block[0], mx = block[0];
        for (const auto& v : block) {
            mn.x = std::fmin(mn.x, v.x); mn.y = std::fmin(mn.y, v.y); mn.z = std::fmin(mn.z, v.z);
            mx.x = std::fmax(mx.x, v.x); mx.y = std::fmax(mx.y, v.y); mx.z = std::fmax(mx.z, v.z);
        }
        PivotRule r = rules[p];
        mesh.pivot = { mn.x + (mx.x - mn.x) * r.fx,
                       mn.y + (mx.y - mn.y) * r.fy,
                       mn.z + (mx.z - mn.z) * r.fz };

        // Store vertices in the LOCAL frame of the joint
        mesh.vertices.reserve(8);
        for (const auto& v : block) mesh.vertices.push_back(v - mesh.pivot);

        // Faces: indices were 1-based into the global vertex list; remap to
        // 0-based local indices.
        int base = static_cast<int>(p * 8) + 1; // first global index in block
        for (const auto& tri : allFaces) {
            bool inBlock = true;
            int local[3];
            for (int k = 0; k < 3; ++k) {
                int g = tri[k];
                int rel = g - base;
                if (rel < 0 || rel >= 8) { inBlock = false; break; }
                local[k] = rel;
            }
            if (inBlock) mesh.faces.push_back({local[0], local[1], local[2]});
        }

        outMeshes.push_back(std::move(mesh));
    }
    return true;
}

#endif // OBJ_LOADER_H
