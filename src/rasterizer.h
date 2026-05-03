// rasterizer.h

#ifndef RASTERIZER_H
#define RASTERIZER_H

#include "math_utils.h"

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <array>
#include <algorithm>
#include <string>

struct Framebuffer {
    int width{0}, height{0};
    std::vector<std::array<uint8_t, 3>> pixels; // row-major, top-down
    std::vector<float> depth;  // per-pixel NDC z, +1 = far

    void resize(int w, int h) {
        width = w; height = h;
        pixels.assign(w * h, {0, 0, 0});
        depth.assign(w * h, 1.f);
    }

    void clear(uint8_t r, uint8_t g, uint8_t b) {
        for (auto& p : pixels) p = {r, g, b};
        std::fill(depth.begin(), depth.end(), 1.f);
    }

    // Write Portable Pixmap (PPM P6).  See Poskanzer's netpbm documentation.
    bool writePPM(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "P6\n%d %d\n255\n", width, height);
        for (const auto& px : pixels) std::fwrite(px.data(), 1, 3, f);
        std::fclose(f);
        return true;
    }
};

// signed area of the triangle formed by the
// directed edge (a -> b) and point p; sign indicates which side p is on.
inline float edge(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Rasterise a flat-shaded triangle with perspective-correct-ish depth.  We
// interpolate the NDC-z linearly because our primitives are small and
// axis-aligned; for this assignment the accuracy is ample.
inline void drawTriangle(Framebuffer& fb, const Projected& p0, const Projected& p1, const Projected& p2, std::array<uint8_t, 3> rgb)
{
    if (!p0.visible || !p1.visible || !p2.visible) return;

    // Backface culling: OBJ front faces wind CCW in 3D (right-hand rule).  After
    // perspective projection the NDC Y-axis is flipped to screen-space Y-down,
    // which reverses the winding sense: a CCW world front face becomes CW in
    // screen space, giving a NEGATIVE signed area from the edge() formula.
    // We therefore keep triangles with area2 < 0 (CW in screen = front face)
    // and cull area2 >= 0 (CCW in screen = back face).
    float area2 = edge(p0.sx, p0.sy, p1.sx, p1.sy, p2.sx, p2.sy);
    if (area2 >= 0.f) return;   // cull back faces (CCW in screen after Y-flip)
    float invArea = 1.f / area2;  // negative; dividing negative w also negates

    // Compute the triangle's pixel bounding box, clipped to the framebuffer.
    int minX = static_cast<int>(std::floor(std::min({p0.sx, p1.sx, p2.sx})));
    int maxX = static_cast<int>(std::ceil (std::max({p0.sx, p1.sx, p2.sx})));
    int minY = static_cast<int>(std::floor(std::min({p0.sy, p1.sy, p2.sy})));
    int maxY = static_cast<int>(std::ceil (std::max({p0.sy, p1.sy, p2.sy})));
    minX = std::max(0, minX);
    minY = std::max(0, minY);
    maxX = std::min(fb.width  - 1, maxX);
    maxY = std::min(fb.height - 1, maxY);
    if (maxX < minX || maxY < minY) return;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            float px = x + 0.5f, py = y + 0.5f;
            float w0 = edge(p1.sx, p1.sy, p2.sx, p2.sy, px, py);
            float w1 = edge(p2.sx, p2.sy, p0.sx, p0.sy, px, py);
            float w2 = edge(p0.sx, p0.sy, p1.sx, p1.sy, px, py);
            // Inside test: all barycentrics non-negative.  Using the same
            // winding sign as the culling test above.
            // For CW triangles (area2 < 0) interior points have all-negative
            // edge values; dividing by the negative invArea yields positive
            // barycentric coordinates automatically.
            if (w0 > 0.f || w1 > 0.f || w2 > 0.f) continue;
            w0 *= invArea; w1 *= invArea; w2 *= invArea;

            float z = p0.depth * w0 + p1.depth * w1 + p2.depth * w2;
            int idx = y * fb.width + x;
            if (z < fb.depth[idx]) {
                fb.depth[idx] = z;
                fb.pixels[idx] = rgb;
            }
        }
    }
}

// Flat lighting: a key directional light plus a softer "fill" light from the
// opposite hemisphere, plus a constant ambient floor.  A key/fill setup is
// the three-point-lighting convention inherited from traditional cinema
// and is standard practice in real-time rendering for giving every
// surface *some* non-ambient response, so white materials (the belly, the
// ground) don't collapse to grey on the shadow side.  The two-lobe form
// N.L_key + 0.35*N.L_fill is the cheapest possible approximation to an
// environment-light integral.
inline std::array<uint8_t, 3> shadeFlat(const Vec3& normalWorld, const Vec3& lightDir, const std::array<float, 3>& baseRgb)
{
    Vec3 n     = normalWorld.normalized();
    Vec3 lKey  = (-lightDir).normalized();
    // Fill light: roughly opposite to the key, biased upward so sky-facing
    // surfaces always receive a little light.
    Vec3 lFill = Vec3{ -lKey.x * 0.5f, 0.4f, -lKey.z * 0.5f }.normalized();

    float key  = std::fmax(0.f, n.dot(lKey));
    float fill = std::fmax(0.f, n.dot(lFill));
    float intensity = 0.22f + 0.62f * key + 0.26f * fill;  // ambient + key + fill
    if (intensity > 1.f) intensity = 1.f;

    auto clamp8 = [](float v) -> uint8_t {
        v = std::fmin(1.f, std::fmax(0.f, v));
        return static_cast<uint8_t>(v * 255.f + 0.5f);
    };
    return { clamp8(baseRgb[0] * intensity), clamp8(baseRgb[1] * intensity), clamp8(baseRgb[2] * intensity) };
}

#endif // RASTERIZER_H
