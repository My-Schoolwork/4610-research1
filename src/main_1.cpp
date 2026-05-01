// main_1.cpp - Real-time SFML viewer for the Cubism penguin walk cycle.
// Showcases the full walk cycle animation with joint ROM overlay.
//
// Controls:
//   Space       - pause / resume
//   Left/Right  - slow down / speed up animation (0.25x to 4x)
//   R           - reset to t=0
//   J           - toggle joint ROM arc overlay
//   Esc / Q     - quit

#include "math_utils.h"
#include "obj_loader.h"
#include "skeleton.h"
#include "rasterizer.h"
#include "joint_rom_ui.h"

#include <SFML/Graphics.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

// ---------------------------------------------------------------------------
// Render configuration
// ---------------------------------------------------------------------------
struct Config {
    int   width     = 960;
    int   height    = 540;
    Vec3  camOffset = { -2.6f, 1.1f, 2.6f };
    Vec3  lightDir  = Vec3{ -0.35f, -0.75f, -0.55f }.normalized();
    std::array<uint8_t, 3> bgTop    = { 150, 190, 230 };
    std::array<uint8_t, 3> bgBottom = {  35,  50,  70 };
};

// ---------------------------------------------------------------------------
// Gradient background
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ice ground plane
// ---------------------------------------------------------------------------
static void drawGround(Framebuffer& fb,
                       const Mat4& viewProj,
                       float centerZ,
                       const Vec3& lightDir)
{
    const float size = 40.f;
    const float y    = -0.82f;
    const int   N    = 24;
    const float step = size / N;

    Vec3  n         = { 0, 1, 0 };
    float lambert   = std::fmax(0.f, n.dot((-lightDir).normalized()));
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

            drawTriangle(fb, pa, pb, pc, rgb);
            drawTriangle(fb, pa, pc, pd, rgb);
        }
    }
}

// ---------------------------------------------------------------------------
// Render one posed frame
// ---------------------------------------------------------------------------
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

    for (int j = 0; j < J_COUNT; ++j) {
        int mi = anim.sk.joints[j].meshIndex;
        if (mi < 0) continue;
        const Mesh& mesh = meshes[mi];
        const Mat4& W    = anim.sk.world[j];

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

// ---------------------------------------------------------------------------
// Upload framebuffer to SFML texture
// ---------------------------------------------------------------------------
static void uploadToTexture(const Framebuffer& fb, sf::Texture& tex)
{
    const int n = fb.width * fb.height;
    static std::vector<sf::Uint8> rgba;
    rgba.resize(n * 4);
    for (int i = 0; i < n; ++i) {
        rgba[i * 4 + 0] = fb.pixels[i][0];
        rgba[i * 4 + 1] = fb.pixels[i][1];
        rgba[i * 4 + 2] = fb.pixels[i][2];
        rgba[i * 4 + 3] = 255;
    }
    tex.update(rgba.data());
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------
static void drawHUD(sf::RenderWindow& win,
                    const sf::Font& font,
                    float simTime,
                    float speedMul,
                    bool paused,
                    bool showROM)
{
    sf::RectangleShape bar(sf::Vector2f(static_cast<float>(win.getSize().x), 36.f));
    bar.setPosition(0.f, static_cast<float>(win.getSize().y) - 36.f);
    bar.setFillColor(sf::Color(0, 0, 0, 140));
    win.draw(bar);

    char buf[192];
    std::snprintf(buf, sizeof(buf),
        "%s  t=%.2fs  speed=%.2fx    [Space] pause  [</> arrows] speed  [R] reset  [J] ROM=%s  [Esc] quit",
        paused ? "  PAUSED" : "PLAYING",
        simTime, speedMul,
        showROM ? "ON" : "OFF");

    sf::Text hud;
    hud.setFont(font);
    hud.setString(buf);
    hud.setCharacterSize(14);
    hud.setFillColor(sf::Color(220, 220, 220, 230));
    hud.setPosition(10.f, static_cast<float>(win.getSize().y) - 28.f);
    win.draw(hud);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <penguin.obj>\n", argv[0]);
        return 1;
    }

    std::vector<Mesh> meshes;
    if (!loadPenguinObj(argv[1], meshes)) return 2;
    std::printf("Loaded %zu meshes from %s\n", meshes.size(), argv[1]);

    Config cfg;

    sf::RenderWindow window(
        sf::VideoMode(cfg.width, cfg.height),
        "Penguin Walk Cycle - Showcase",
        sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    sf::Texture frameTex;
    frameTex.create(cfg.width, cfg.height);
    sf::Sprite frameSprite(frameTex);

    sf::Font font;
    bool hasFont = false;
    for (const char* p : {
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "C:/Windows/Fonts/consola.ttf"}) {
        if (font.loadFromFile(p)) { hasFont = true; break; }
    }

    Framebuffer fb;
    fb.resize(cfg.width, cfg.height);

    Animator anim;
    anim.init(meshes);

    WalkParams wp;
    wp.speedRatio = 1.f;   // full walk speed for the showcase

    std::vector<JointROM> jointROMs = computeJointROMs(wp);

    Mat4 proj = perspective(45.f * 3.14159265f / 180.f,
                            static_cast<float>(cfg.width) / cfg.height,
                            0.1f, 100.f);

    sf::Clock clock;
    float simTime  = 0.f;
    float speedMul = 1.f;
    bool  paused   = false;
    bool  showROM  = false;

    while (window.isOpen()) {
        // ---- Events --------------------------------------------------------
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed) {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                case sf::Keyboard::Q:
                    window.close();
                    break;
                case sf::Keyboard::Space:
                    paused = !paused;
                    break;
                case sf::Keyboard::R:
                    simTime = 0.f;
                    clock.restart();
                    break;
                case sf::Keyboard::Right:
                    speedMul = std::min(speedMul * 1.25f, 4.f);
                    break;
                case sf::Keyboard::Left:
                    speedMul = std::max(speedMul / 1.25f, 0.25f);
                    break;
                case sf::Keyboard::J:
                    showROM = !showROM;
                    break;
                default: break;
                }
            }
        }

        // ---- Advance simulation time ---------------------------------------
        float dt = clock.restart().asSeconds();
        dt = std::min(dt, 0.05f);
        if (!paused)
            simTime += dt * speedMul;

        // idleTime always mirrors simTime so beak chatter and breathing play
        wp.idleTime = simTime;

        // ---- Pose & render -------------------------------------------------
        anim.pose(simTime, wp);

        float penguinZ = wp.forwardSpeed * simTime;
        Vec3  target   = { 0.f, 0.3f, penguinZ };
        Vec3  eye      = { cfg.camOffset.x,
                           cfg.camOffset.y,
                           penguinZ + cfg.camOffset.z };
        Mat4  view     = lookAt(eye, target, {0, 1, 0});
        Mat4  vp       = proj * view;

        renderFrame(fb, meshes, anim, vp, cfg.lightDir, penguinZ, cfg);

        uploadToTexture(fb, frameTex);
        window.clear();
        window.draw(frameSprite);

        if (showROM) {
            drawJointROMArcs(window, vp, cfg.width, cfg.height,
                             anim.sk, jointROMs);
            if (hasFont)
                drawROMPanel(window, font, anim.sk, jointROMs);
        }

        if (hasFont)
            drawHUD(window, font, simTime, speedMul, paused, showROM);

        window.display();
    }

    return 0;
}