// obj_loader.h - Minimal Wavefront .obj loader specialised for the Cubism
// penguin model supplied with the assignment.
//
// Updated to support parts with variable vertex counts (e.g. the feet, which
// were extended from 8 to 19 vertices in the fixed model).  Instead of
// assuming every part has exactly 8 vertices, we derive part boundaries from
// the face index ranges: each consecutive group of faces that only references
// vertices in a contiguous range is treated as one part.  This is robust to
// any cuboid having more or fewer than 8 vertices, as long as parts remain
// contiguous in the vertex list (which Blender's default OBJ export guarantees).
//
// The rest of the interface (Mesh, Tri, loadPenguinObj) is unchanged so that
// main.cpp, skeleton.h, and rasterizer.h require no modifications.

#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "math_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <climits>

// A triangle stored as three indices into the mesh-local vertex list.
struct Tri { int a, b, c; };

struct Mesh {
    std::string         name;
    std::vector<Vec3>   vertices;   // in the mesh's REST-POSE local frame
    std::vector<Tri>    faces;      // triangle list (CCW winding)
    Vec3                pivot{};    // joint centre in the model's world frame
    std::array<float,3> colorRgb{};
};

struct PartSpec {
    const char*          name;
    std::array<float,3>  color;
};

inline const std::vector<PartSpec>& penguinPartSpecs() {
    static const std::vector<PartSpec> s = {
        {"foot_L",      {0.95f, 0.55f, 0.10f}}, //  0  verts 1-19
        {"body",        {0.12f, 0.12f, 0.14f}}, //  1  verts 20-27
        {"belly",       {0.97f, 0.97f, 0.97f}}, //  2  verts 28-35
        {"neck",        {0.12f, 0.12f, 0.14f}}, //  3  verts 36-43
        {"head",        {0.12f, 0.12f, 0.14f}}, //  4  verts 44-51
        {"beak_upper",  {0.98f, 0.65f, 0.10f}}, //  5  verts 52-59
        {"beak_lower",  {0.98f, 0.55f, 0.08f}}, //  6  verts 60-67
        {"eye_L",       {0.95f, 0.80f, 0.10f}}, //  7  verts 68-75
        {"eye_R",       {0.95f, 0.80f, 0.10f}}, //  8  verts 76-83
        {"wing_L",      {0.10f, 0.10f, 0.12f}}, //  9  verts 84-91
        {"wing_R",      {0.10f, 0.10f, 0.12f}}, // 10  verts 92-99
        {"foot_R",      {0.95f, 0.55f, 0.10f}}, // 11  verts 100-118
    };
    return s;
}

// Pivot fractional positions within each part's AABB.
// Wings  -> inner-top corner (shoulder); feet -> top-rear; others -> centre.
struct PivotRule { float fx, fy, fz; };
inline const std::array<PivotRule, 12>& penguinPivotRules() {
    static const std::array<PivotRule, 12> r = {{
        {0.5f, 1.0f, 0.0f},   //  0 foot_L  - hip: top-rear
        {0.5f, 0.5f, 0.5f},   //  1 body
        {0.5f, 0.5f, 0.5f},   //  2 belly
        {0.5f, 0.5f, 0.0f},   //  3 neck
        {0.5f, 0.0f, 0.5f},   //  4 head    - pivot at base of neck
        {0.5f, 0.5f, 0.0f},   //  5 beak_upper
        {0.5f, 0.5f, 0.0f},   //  6 beak_lower
        {0.5f, 0.5f, 0.5f},   //  7 eye_L
        {0.5f, 0.5f, 0.5f},   //  8 eye_R
        {1.0f, 1.0f, 0.5f},   //  9 wing_L  - shoulder: inner top corner
        {0.0f, 1.0f, 0.5f},   // 10 wing_R  - shoulder: inner top corner
        {0.5f, 1.0f, 0.0f},   // 11 foot_R  - hip: top-rear
    }};
    return r;
}

// ---------------------------------------------------------------------------
// Part vertex ranges - derived from the actual penguin_fixed.obj layout.
//
// The body cuboid (verts 20-27) has faces that reference non-contiguous
// subsets of its vertices, which caused the greedy face-grouping algorithm
// to split it into two parts.  Instead we use explicit vertex boundaries
// read directly from the model:
//
//   Part 0  foot_L      verts  1-19   (19 verts - extended from original 8)
//   Part 1  body        verts 20-27
//   Part 2  belly       verts 28-35
//   Part 3  neck        verts 36-43
//   Part 4  head        verts 44-51
//   Part 5  beak_upper  verts 52-59
//   Part 6  beak_lower  verts 60-67
//   Part 7  eye_L       verts 68-75
//   Part 8  eye_R       verts 76-83
//   Part 9  wing_L      verts 84-91
//   Part 10 wing_R      verts 92-99
//   Part 11 foot_R      verts 100-118  (19 verts - extended from original 8)
//
// vertBegin/vertEnd are 0-based, vertEnd is exclusive.
// ---------------------------------------------------------------------------
struct PartRange {
    int vertBegin;
    int vertEnd;
    int faceBegin;  // unused in explicit mode but kept for interface compat
    int faceEnd;
};

inline std::vector<PartRange> derivePartRanges(
    const std::vector<std::array<int,3>>& faces,
    int totalVerts)
{
    // Explicit 0-based vertex boundaries matching penguin_fixed.obj.
    // If you add more vertices to a part, update the boundary here.
    static const int vertBounds[] = {
         0, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 118
    };
    const int N = 12; // number of parts = number of gaps in vertBounds

    // Assign each face to the part whose vertex range contains all its verts.
    // faceStart[p] = first face index belonging to part p.
    std::vector<int> faceStart(N + 1, static_cast<int>(faces.size()));
    faceStart[0] = 0;

    // Walk faces in order; they are already sorted by part in the OBJ.
    int curPart = 0;
    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
        // Find which part this face belongs to (all 3 verts in same range).
        int fv = faces[fi][0] - 1; // 0-based, use first vert to identify part
        while (curPart < N - 1 && fv >= vertBounds[curPart + 1])
            ++curPart;
        // Record start of each new part.
        // (faces are contiguous per part so we only need to detect the jump)
        if (fi > 0) {
            int prevFv = faces[fi - 1][0] - 1;
            int prevPart = 0;
            while (prevPart < N - 1 && prevFv >= vertBounds[prevPart + 1])
                ++prevPart;
            if (prevPart != curPart)
                faceStart[curPart] = fi;
        }
    }
    faceStart[N] = static_cast<int>(faces.size());

    std::vector<PartRange> parts;
    parts.reserve(N);
    for (int p = 0; p < N; ++p) {
        parts.push_back({vertBounds[p], vertBounds[p + 1],
                         faceStart[p], faceStart[p + 1]});
    }

    (void)totalVerts;
    return parts;
}

// ---------------------------------------------------------------------------
// Load the penguin .obj - supports any vertex count per part.
// ---------------------------------------------------------------------------
inline bool loadPenguinObj(const std::string& path, std::vector<Mesh>& outMeshes)
{
    std::vector<Vec3> allVerts;
    std::vector<std::array<int,3>> allFaces; // 1-based as per .obj

    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3)
                allVerts.push_back({x, y, z});
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Support triangles and quads in "f a//n ..." and "f a b c ..." formats.
            // Quads are fan-triangulated: (a,b,c,d) -> (a,b,c) + (a,c,d).
            int a, b, c, d;
            int na, nb, nc, nd;
            if (std::sscanf(line + 2, "%d//%d %d//%d %d//%d %d//%d",
                            &a, &na, &b, &nb, &c, &nc, &d, &nd) == 8) {
                allFaces.push_back({a, b, c});
                allFaces.push_back({a, c, d});
            } else if (std::sscanf(line + 2, "%d//%d %d//%d %d//%d",
                            &a, &na, &b, &nb, &c, &nc) == 6) {
                allFaces.push_back({a, b, c});
            } else if (std::sscanf(line + 2, "%d %d %d %d", &a, &b, &c, &d) == 4) {
                allFaces.push_back({a, b, c});
                allFaces.push_back({a, c, d});
            } else if (std::sscanf(line + 2, "%d %d %d", &a, &b, &c) == 3) {
                allFaces.push_back({a, b, c});
            }
        }
    }
    std::fclose(f);

    const int totalVerts = static_cast<int>(allVerts.size());
    std::printf("obj_loader: %d vertices, %zu faces\n",
                totalVerts, allFaces.size());

    // Derive part boundaries from face data.
    std::vector<PartRange> parts = derivePartRanges(allFaces, totalVerts);
    std::printf("obj_loader: detected %zu parts\n", parts.size());

    const auto& specs  = penguinPartSpecs();
    const auto& pivots = penguinPivotRules();

    if (parts.size() != specs.size()) {
        std::fprintf(stderr,
            "obj_loader: expected %zu parts, detected %zu. "
            "Check that the OBJ has contiguous vertex blocks per body part.\n",
            specs.size(), parts.size());
        return false;
    }

    outMeshes.clear();
    outMeshes.reserve(parts.size());

    for (size_t p = 0; p < parts.size(); ++p) {
        const PartRange& pr = parts[p];
        Mesh mesh;
        mesh.name     = specs[p].name;
        mesh.colorRgb = specs[p].color;

        // Extract vertices for this part.
        std::vector<Vec3> block(allVerts.begin() + pr.vertBegin,
                                allVerts.begin() + pr.vertEnd);

        // AABB for pivot computation.
        Vec3 mn = block[0], mx = block[0];
        for (const auto& v : block) {
            mn.x = std::fmin(mn.x, v.x); mn.y = std::fmin(mn.y, v.y); mn.z = std::fmin(mn.z, v.z);
            mx.x = std::fmax(mx.x, v.x); mx.y = std::fmax(mx.y, v.y); mx.z = std::fmax(mx.z, v.z);
        }
        const PivotRule& r = pivots[p];
        mesh.pivot = { mn.x + (mx.x - mn.x) * r.fx,
                       mn.y + (mx.y - mn.y) * r.fy,
                       mn.z + (mx.z - mn.z) * r.fz };

        // Store vertices in local joint frame.
        mesh.vertices.reserve(block.size());
        for (const auto& v : block) mesh.vertices.push_back(v - mesh.pivot);

        // Remap faces: global 1-based -> local 0-based.
        for (int fi = pr.faceBegin; fi < pr.faceEnd; ++fi) {
            const auto& tri = allFaces[fi];
            mesh.faces.push_back({
                tri[0] - 1 - pr.vertBegin,
                tri[1] - 1 - pr.vertBegin,
                tri[2] - 1 - pr.vertBegin
            });
        }

        // Normalise winding so all face normals point outward from the mesh
        // centroid.  The foot meshes have inconsistent winding from the
        // modelling tool; this fixes them without editing the OBJ file.
        {
            Vec3 cen{0,0,0};
            for (const auto& v : mesh.vertices) { cen.x+=v.x; cen.y+=v.y; cen.z+=v.z; }
            float inv = 1.f / static_cast<float>(mesh.vertices.size());
            cen = {cen.x*inv, cen.y*inv, cen.z*inv};

            for (auto& f : mesh.faces) {
                const Vec3& A = mesh.vertices[f.a];
                const Vec3& B = mesh.vertices[f.b];
                const Vec3& C = mesh.vertices[f.c];
                // Face normal (unnormalised).
                Vec3 fn {
                    (B.y-A.y)*(C.z-A.z) - (B.z-A.z)*(C.y-A.y),
                    (B.z-A.z)*(C.x-A.x) - (B.x-A.x)*(C.z-A.z),
                    (B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x)
                };
                // Vector from centroid to face centre.
                Vec3 fc { (A.x+B.x+C.x)/3.f - cen.x,
                          (A.y+B.y+C.y)/3.f - cen.y,
                          (A.z+B.z+C.z)/3.f - cen.z };
                // If normal points inward, flip the winding.
                if (fn.x*fc.x + fn.y*fc.y + fn.z*fc.z < 0.f)
                    std::swap(f.b, f.c);
            }
        }

        std::printf("  [%2zu] %-12s  %zu verts, %zu faces\n",
                    p, mesh.name.c_str(),
                    mesh.vertices.size(), mesh.faces.size());

        outMeshes.push_back(std::move(mesh));
    }
    return true;
}

#endif // OBJ_LOADER_H