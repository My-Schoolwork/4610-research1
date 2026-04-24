// main.cpp - Task 1 driver.  Loads the Cubism penguin, builds a 12-joint
// skeleton around it, drives a procedural walk cycle, and renders the result
// to a sequence of PPM frames that an external tool (ffmpeg) then stitches
// into an MP4.
//
// Build (see CMakeLists.txt / build.sh):
//   g++ -std=c++17 -O2 src/main.cpp -o build/penguin_walk
//
// Run:
//   ./penguin_walk assets/penguin.obj build/frames
//
// All of the interesting computation - the scene graph, the walk-cycle
// phase functions, and the rasteriser - is in the accompanying headers.
// This file is essentially glue: read args, set up camera, loop over time,
// call applyWalkPose, draw, write.

#include "math_utils.h"
#include "obj_loader.h"
#include "skeleton.h"
#include "rasterizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <filesystem>

namespace fs = std::filesystem;

// Render configuration.  Chosen for a 15-second clip at 30 fps -> 450 frames.
struct Config {
    int   width     = 960;
    int   height    = 540;
    int   fps       = 30;
    float duration  = 15.0f;  // seconds
    // Camera sits on the penguin's FORWARD-LEFT quarter, slightly above eye
    // height, and tracks it.  This 3/4 view is the standard "character
    // showcase" framing (see e.g. Thomas & Johnston, Disney Animation: The
    // Illusion of Life, 1981, Ch. 4) - it reveals the face, the wing swing,
    // AND the side-to-side waddle at the same time.  The penguin walks along
    // +Z, so "forward-left" = (+X, -Z(ish) ahead of it).
    Vec3  camOffset = { -2.6f, 1.1f, 2.6f }; // relative to penguin pelvis
    // Directional light pointing down-and-slightly-back so the belly (which
    // faces +Z toward the camera) receives a clear Lambertian fill and the
    // head, wings, and feet keep some shadowed faces for form definition.
    // lightDir is the direction the light travels; we dot with (-lightDir).
    Vec3  lightDir  = Vec3{ -0.35f, -0.75f, -0.55f }.normalized();
    std::array<uint8_t, 3> bgTop    = { 150, 190, 230 };
    std::array<uint8_t, 3> bgBottom = {  35,  50,  70 };
};

// Vertical gradient background - gives the scene some visual interest
// without the cost of a proper skybox.
static void clearGradient(Framebuffer& fb,
                          std::array<uint8_t, 3> top,
                          std::array<uint8_t, 3> bot)
{
    for (int y = 0; y < fb.height; ++y) {
        float t = static_cast<float>(y) / std::max(1, fb.height - 1);
        std::array<uint8_t, 3> c = {
            static_cast<uint8_t>(top[0] * (1 - t) + bot[0] * t),
            static_cast<uint8_t>(top[1] * (1 - t) + bot[1] * t),
            static_cast<uint8_t>(top[2] * (1 - t) + bot[2] * t)};
        for (int x = 0; x < fb.width; ++x) {
            fb.pixels[y * fb.width + x] = c;
            fb.depth [y * fb.width + x] = 1.f;
        }
    }
}

// A simple textured "ice" ground plane drawn as two triangles with a tiled
// checker pattern.  It gives the eye a reference frame so the forward motion
// of the penguin is unambiguous - without it the viewer can't tell whether
// the animal or the camera is moving.
static void drawGround(Framebuffer& fb,
                       const Mat4& viewProj,
                       float centerZ,
                       const Vec3& lightDir)
{
    const float size = 40.f;
    const float y    = -0.82f;       // ground height (matches foot bottoms)
    // Grid of 24 x 24 quads centered on centerZ; each quad gets a shade based
    // on (ix + iz) & 1, with a subtle Lambert response against the light.
    const int   N = 24;
    const float step = size / N;
    // Plane normal points +Y.
    Vec3 n = { 0, 1, 0 };
    float lambert = std::fmax(0.f, n.dot((-lightDir).normalized()));
    float intensity = 0.45f + 0.55f * lambert;

    for (int iz = 0; iz < N; ++iz) {
        for (int ix = 0; ix < N; ++ix) {
            float x0 = -size * 0.5f + ix * step;
            float z0 = centerZ - size * 0.5f + iz * step;
            float x1 = x0 + step, z1 = z0 + step;
            Vec3 a{x0, y, z0}, b{x1, y, z0}, c{x1, y, z1}, d{x0, y, z1};
            Projected pa = projectPoint(viewProj, a, fb.width, fb.height);
            Projected pb = projectPoint(viewProj, b, fb.width, fb.height);
            Projected pc = projectPoint(viewProj, c, fb.width, fb.height);
            Projected pd = projectPoint(viewProj, d, fb.width, fb.height);

            bool dark = ((ix + iz) & 1) != 0;
            std::array<float, 3> base = dark
                ? std::array<float,3>{0.70f, 0.78f, 0.88f}
                : std::array<float,3>{0.86f, 0.92f, 0.98f};
            std::array<uint8_t, 3> rgb = {
                static_cast<uint8_t>(base[0] * intensity * 255.f),
                static_cast<uint8_t>(base[1] * intensity * 255.f),
                static_cast<uint8_t>(base[2] * intensity * 255.f)};
            // Two triangles (a,b,c) and (a,c,d).  Winding chosen to match the
            // rasteriser's CCW-in-screen convention when viewed from above.
            drawTriangle(fb, pa, pb, pc, rgb);
            drawTriangle(fb, pa, pc, pd, rgb);
        }
    }
}

// Render one posed frame.
static void renderFrame(Framebuffer& fb,
                        const std::vector<Mesh>& meshes,
                        const Animator& anim,
                        const Mat4& viewProj,
                        const Vec3& lightDir,
                        float penguinZ,
                        const Config& cfg)
{
    clearGradient(fb, cfg.bgTop, cfg.bgBottom);
    drawGround(fb, viewProj, penguinZ, lightDir);

    // For each joint that owns a mesh, transform the mesh's local vertices
    // into world space, project them, and rasterise every triangle.  We
    // compute each triangle's face normal in world space for Lambertian
    // shading - cheaper and visually cleaner than per-vertex normals for a
    // flat-shaded cubist model.
    for (int j = 0; j < J_COUNT; ++j) {
        int mi = anim.sk.joints[j].meshIndex;
        if (mi < 0) continue;
        const Mesh& mesh = meshes[mi];
        const Mat4& W    = anim.sk.world[j];

        // Transform vertices once.
        std::vector<Vec3>      ws(mesh.vertices.size());
        std::vector<Projected> ps(mesh.vertices.size());
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            ws[i] = W.transformPoint(mesh.vertices[i]);
            ps[i] = projectPoint(viewProj, ws[i], fb.width, fb.height);
        }

        for (const auto& tri : mesh.faces) {
            const Vec3& A = ws[tri.a];
            const Vec3& B = ws[tri.b];
            const Vec3& C = ws[tri.c];
            Vec3 n = (B - A).cross(C - A).normalized();
            auto rgb = shadeFlat(n, lightDir, mesh.colorRgb);
            drawTriangle(fb, ps[tri.a], ps[tri.b], ps[tri.c], rgb);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <penguin.obj> <frames_output_dir>\n", argv[0]);
        return 1;
    }
    std::string objPath   = argv[1];
    std::string framesDir = argv[2];
    fs::create_directories(framesDir);

    std::vector<Mesh> meshes;
    if (!loadPenguinObj(objPath, meshes)) return 2;
    std::printf("Loaded %zu meshes from %s\n", meshes.size(), objPath.c_str());
    for (size_t i = 0; i < meshes.size(); ++i) {
        std::printf("  [%2zu] %-12s  %zu verts, %zu faces, pivot=(%+.3f,%+.3f,%+.3f)\n",
                    i, meshes[i].name.c_str(),
                    meshes[i].vertices.size(), meshes[i].faces.size(),
                    meshes[i].pivot.x, meshes[i].pivot.y, meshes[i].pivot.z);
    }

    Config cfg;
    Framebuffer fb;
    fb.resize(cfg.width, cfg.height);

    Animator anim;
    anim.init(meshes);

    WalkParams wp; // default penguin-like parameters

    const int totalFrames = static_cast<int>(cfg.duration * cfg.fps);
    std::printf("Rendering %d frames at %dx%d, %d fps (%.1f s)\n",
                totalFrames, cfg.width, cfg.height, cfg.fps, cfg.duration);

    Mat4 proj = perspective(45.f * 3.14159265f / 180.f,
                            static_cast<float>(cfg.width) / cfg.height,
                            0.1f, 100.f);

    for (int f = 0; f < totalFrames; ++f) {
        float t = static_cast<float>(f) / cfg.fps;
        anim.pose(t, wp);

        // Camera tracks the penguin so it stays centred in the frame.
        float penguinZ = wp.forwardSpeed * t;      // matches root offset
        // We want the camera slightly behind and above, looking forward.
        Vec3 target = { 0.f, 0.3f, penguinZ };
        Vec3 eye    = { cfg.camOffset.x,
                        cfg.camOffset.y,
                        penguinZ + cfg.camOffset.z };
        Mat4 view   = lookAt(eye, target, {0, 1, 0});
        Mat4 vp     = proj * view;

        renderFrame(fb, meshes, anim, vp, cfg.lightDir, penguinZ, cfg);

        char name[256];
        std::snprintf(name, sizeof(name), "%s/frame_%04d.ppm",
                      framesDir.c_str(), f);
        fb.writePPM(name);

        if (f % 30 == 0) {
            std::printf("  frame %4d / %4d\n", f, totalFrames);
            std::fflush(stdout);
        }
    }
    std::printf("Done: %d frames written to %s\n", totalFrames, framesDir.c_str());
    return 0;
}
