// math_utils.h - Minimal linear-algebra primitives for the animation project.

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <array>
#include <cstdio>

struct Vec3 {
    float x{0.f}, y{0.f}, z{0.f};
    Vec3() = default;
    constexpr Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s,   y * s,   z * s};   }
    Vec3 operator-() const { return {-x, -y, -z}; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3  cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float length() const { return std::sqrt(dot(*this)); }
    Vec3  normalized() const {
        float n = length();
        return (n > 1e-8f) ? Vec3{x / n, y / n, z / n} : Vec3{0, 0, 0};
    }
};

// Row-major 4x4 matrix.  m[row][col].
struct Mat4 {
    std::array<std::array<float, 4>, 4> m{};

    static Mat4 identity() {
        Mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.f;
        return r;
    }

    static Mat4 translation(float tx, float ty, float tz) {
        Mat4 r = identity();
        r.m[0][3] = tx;  r.m[1][3] = ty;  r.m[2][3] = tz;
        return r;
    }
    static Mat4 translation(const Vec3& t) { return translation(t.x, t.y, t.z); }

    static Mat4 scale(float s) {
        Mat4 r = identity();
        r.m[0][0] = r.m[1][1] = r.m[2][2] = s;
        return r;
    }

    // Right-handed rotations.  Angle in radians.
    static Mat4 rotationX(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[1][1] =  c; r.m[1][2] = -s;
        r.m[2][1] =  s; r.m[2][2] =  c;
        return r;
    }
    static Mat4 rotationY(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0][0] =  c; r.m[0][2] =  s;
        r.m[2][0] = -s; r.m[2][2] =  c;
        return r;
    }
    static Mat4 rotationZ(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0][0] =  c; r.m[0][1] = -s;
        r.m[1][0] =  s; r.m[1][1] =  c;
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
                for (int k = 0; k < 4; ++k) s += m[i][k] * o.m[k][j];
                r.m[i][j] = s;
            }
        return r;
    }

    // Transform a point (w = 1).
    Vec3 transformPoint(const Vec3& v) const {
        float x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3];
        float y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3];
        float z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3];
        return {x, y, z};
    }
};

// A perspective projection matrix (OpenGL-style: +X right, +Y up, -Z forward).
// fovY in radians, aspect = width/height.  After projection the perspective
// divide (x/w, y/w, z/w) yields normalised device coordinates in [-1, 1]^3.
inline Mat4 perspective(float fovY, float aspect, float n, float f) {
    Mat4 r;
    float t = std::tan(fovY * 0.5f);
    r.m[0][0] = 1.f / (aspect * t);
    r.m[1][1] = 1.f / t;
    r.m[2][2] = -(f + n) / (f - n);
    r.m[2][3] = -(2.f * f * n) / (f - n);
    r.m[3][2] = -1.f;
    return r;
}

// A simple "look-at" view matrix.  Follows the standard derivation in
// Foley et al. (2013, §7.1) and the OpenGL gluLookAt specification.
inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = (center - eye).normalized();
    Vec3 s = f.cross(up).normalized();
    Vec3 u = s.cross(f);

    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;
    r.m[2][0] =-f.x;  r.m[2][1] =-f.y;  r.m[2][2] =-f.z;
    r.m[0][3] = -s.dot(eye);
    r.m[1][3] = -u.dot(eye);
    r.m[2][3] =  f.dot(eye);
    return r;
}

// Project a world-space point to pixel coordinates plus a depth value.  Used
// by the software rasteriser; returns false if the point lies behind the eye.
struct Projected {
    float sx, sy; // screen-space (pixel) coords
    float depth; // NDC z in [-1, 1]; smaller = closer
    bool  visible; // false if behind the near plane
};

inline Projected projectPoint(const Mat4& mvp, const Vec3& p, int width, int height)
{
    // Full homogeneous transform so we can test w before dividing.
    float x = mvp.m[0][0]*p.x + mvp.m[0][1]*p.y + mvp.m[0][2]*p.z + mvp.m[0][3];
    float y = mvp.m[1][0]*p.x + mvp.m[1][1]*p.y + mvp.m[1][2]*p.z + mvp.m[1][3];
    float z = mvp.m[2][0]*p.x + mvp.m[2][1]*p.y + mvp.m[2][2]*p.z + mvp.m[2][3];
    float w = mvp.m[3][0]*p.x + mvp.m[3][1]*p.y + mvp.m[3][2]*p.z + mvp.m[3][3];

    Projected out{};
    if (w <= 1e-4f) { out.visible = false; return out; }
    float invW = 1.f / w;
    float ndcX = x * invW;
    float ndcY = y * invW;
    float ndcZ = z * invW;

    out.sx = (ndcX * 0.5f + 0.5f) * static_cast<float>(width);
    out.sy = (1.f - (ndcY * 0.5f + 0.5f)) * static_cast<float>(height);
    out.depth = ndcZ;
    out.visible = true;
    return out;
}

#endif // MATH_UTILS_H
