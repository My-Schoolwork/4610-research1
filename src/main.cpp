// main.cpp - Interactive WASD-controlled penguin walk cycle viewer.
//
// Controls:
//   W / S       - walk forward / backward
//   A / D       - turn left / right
//   0           - third-person follow camera (default)
//   1-5         - fixed cameras (Resident Evil style)
//   Space       - pause / resume
//   R           - reset position and orientation
//   Esc / Q     - quit

#include "math_utils.h"
#include "obj_loader.h"
#include "skeleton.h"
#include "rasterizer.h"

#include <SFML/Graphics.hpp>
#include <SFML/Window/Keyboard.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
    float camDist   = 3.8f;
    float camHeight = 1.4f;
    Vec3  lightDir  = Vec3{ -0.35f, -0.75f, -0.55f }.normalized();
    std::array<uint8_t,3> bgTop    = { 150, 190, 230 };
    std::array<uint8_t,3> bgBottom = {  35,  50,  70 };
};

// ---------------------------------------------------------------------------
// Fixed camera - Resident Evil style.
// Eye position is frozen in world space; target tracks the penguin each frame.
// ---------------------------------------------------------------------------
struct FixedCamera {
    const char* name;
    Vec3  eye;    // world-space position, never moves
    float fovY;   // radians
};

static const FixedCamera FIXED_CAMS[] = {
    { "Low front",    {  0.0f, 0.4f,  6.0f }, 50.f * 3.14159265f / 180.f },
    { "High side",    {  6.0f, 4.0f,  0.0f }, 45.f * 3.14159265f / 180.f },
    { "Overhead",     {  0.0f, 9.0f,  1.0f }, 55.f * 3.14159265f / 180.f },
    { "Low rear",     {  0.0f, 0.5f, -6.0f }, 50.f * 3.14159265f / 180.f },
    { "Dramatic 3/4", {  4.0f, 0.2f,  4.0f }, 60.f * 3.14159265f / 180.f },
};
static const int NUM_FIXED_CAMS = static_cast<int>(
    sizeof(FIXED_CAMS) / sizeof(FIXED_CAMS[0]));

// ---------------------------------------------------------------------------
// Player / penguin world state
// ---------------------------------------------------------------------------
struct PlayerState {
    float posX      = 0.f;
    float posZ      = 0.f;
    float yaw       = 0.f;
    float speed     = 0.f;
    float animPhase = 0.f;

    static constexpr float MAX_SPEED = 2.0f;
    static constexpr float ACCEL     = 4.0f;
    static constexpr float DECEL     = 6.0f;
    static constexpr float TURN_RATE = 2.2f;
};

// ---------------------------------------------------------------------------
// Gradient background
// ---------------------------------------------------------------------------
static void clearGradient(Framebuffer& fb,
                          std::array<uint8_t,3> top,
                          std::array<uint8_t,3> bot)
{
    for (int y = 0; y < fb.height; ++y) {
        float t = static_cast<float>(y) / std::max(1, fb.height - 1);
        std::array<uint8_t,3> c = {
            static_cast<uint8_t>(top[0]*(1-t)+bot[0]*t),
            static_cast<uint8_t>(top[1]*(1-t)+bot[1]*t),
            static_cast<uint8_t>(top[2]*(1-t)+bot[2]*t)};
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
                       float centerX, float centerZ,
                       const Vec3& lightDir)
{
    const float size = 60.f;
    const float y    = -0.845f;  // matches foot_L bottom vertex in penguin_fixed.obj
    const int   N    = 30;
    const float step = size / N;
    Vec3  n         = { 0, 1, 0 };
    float lambert   = std::fmax(0.f, n.dot((-lightDir).normalized()));
    float intensity = 0.45f + 0.55f * lambert;

    float snapX = std::floor(centerX / step) * step;
    float snapZ = std::floor(centerZ / step) * step;

    for (int iz = 0; iz < N; ++iz) {
        for (int ix = 0; ix < N; ++ix) {
            float x0 = snapX - size*0.5f + ix*step;
            float z0 = snapZ - size*0.5f + iz*step;
            float x1 = x0+step, z1 = z0+step;
            Vec3 a{x0,y,z0}, b{x1,y,z0}, c{x1,y,z1}, d{x0,y,z1};
            Projected pa = projectPoint(viewProj, a, fb.width, fb.height);
            Projected pb = projectPoint(viewProj, b, fb.width, fb.height);
            Projected pc = projectPoint(viewProj, c, fb.width, fb.height);
            Projected pd = projectPoint(viewProj, d, fb.width, fb.height);

            int tx = static_cast<int>(std::floor(x0/step));
            int tz = static_cast<int>(std::floor(z0/step));
            bool dark = ((tx + tz) & 1) != 0;
            std::array<float,3> base = dark
                ? std::array<float,3>{0.70f, 0.78f, 0.88f}
                : std::array<float,3>{0.86f, 0.92f, 0.98f};
            std::array<uint8_t,3> rgb = {
                static_cast<uint8_t>(base[0]*intensity*255.f),
                static_cast<uint8_t>(base[1]*intensity*255.f),
                static_cast<uint8_t>(base[2]*intensity*255.f)};
            // Ground tiles wind (a,b,c,d) with the normal pointing -Y in 3D.
            // After the NDC→screen Y-flip the corrected culling keeps area2 < 0
            // (front faces), so we reverse the winding here to make the upward-
            // facing surface front-facing.
            drawTriangle(fb, pa, pc, pb, rgb);
            drawTriangle(fb, pa, pd, pc, rgb);
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
                        float penguinX, float penguinZ,
                        const Config& cfg)
{
    clearGradient(fb, cfg.bgTop, cfg.bgBottom);
    drawGround(fb, viewProj, penguinX, penguinZ, lightDir);

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
            Vec3 n = (B-A).cross(C-A).normalized();
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
        rgba[i*4+0] = fb.pixels[i][0];
        rgba[i*4+1] = fb.pixels[i][1];
        rgba[i*4+2] = fb.pixels[i][2];
        rgba[i*4+3] = 255;
    }
    tex.update(rgba.data());
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------
static void drawHUD(sf::RenderWindow& win, const sf::Font& font,
                    const PlayerState& ps, bool paused, int camIndex)
{
    // Bottom bar
    sf::RectangleShape bar(sf::Vector2f(
        static_cast<float>(win.getSize().x), 38.f));
    bar.setPosition(0.f, static_cast<float>(win.getSize().y) - 38.f);
    bar.setFillColor(sf::Color(0,0,0,150));
    win.draw(bar);

    char camName[64];
    if (camIndex < 0)
        std::snprintf(camName, sizeof(camName), "Follow [0]");
    else
        std::snprintf(camName, sizeof(camName), "Cam %d: %s",
                      camIndex + 1, FIXED_CAMS[camIndex].name);

    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "%s | %s | spd=%.2f | [WASD] move  [0-5] cam  [Space] pause  [R] reset  [Esc] quit",
        paused ? "PAUSED" : "PLAYING", camName, ps.speed);

    sf::Text hud;
    hud.setFont(font);
    hud.setString(buf);
    hud.setCharacterSize(13);
    hud.setFillColor(sf::Color(220,220,220,230));
    hud.setPosition(10.f, static_cast<float>(win.getSize().y) - 28.f);
    win.draw(hud);

    // Camera label top-left
    sf::Text label;
    label.setFont(font);
    label.setString(camName);
    label.setCharacterSize(16);
    label.setFillColor(sf::Color(255, 220, 80, 230));
    label.setPosition(10.f, 10.f);
    win.draw(label);
}

// ---------------------------------------------------------------------------
// Root transform
// ---------------------------------------------------------------------------
static Mat4 penguinRootTransform(float posX, float posZ, float yaw)
{
    return Mat4::translation(posX, 0.f, posZ) * Mat4::rotationY(yaw);
}

// ---------------------------------------------------------------------------
// Compute the combined view-projection matrix for the active camera.
//   camIndex == -1  ->  follow camera
//   camIndex 0..4  ->  fixed cameras 1-5
// Fixed cameras have a frozen eye but always look at the penguin.
// ---------------------------------------------------------------------------
static Mat4 computeViewProj(int camIndex, const PlayerState& player,
                            const Config& cfg, float aspect)
{
    Vec3  target = { player.posX, 0.3f, player.posZ };
    Vec3  eye;
    float fovY;

    if (camIndex < 0) {
        float sy = std::sin(player.yaw), cy = std::cos(player.yaw);
        eye  = { player.posX - sy * cfg.camDist,
                 cfg.camHeight,
                 player.posZ - cy * cfg.camDist };
        fovY = 45.f * 3.14159265f / 180.f;
    } else {
        eye  = FIXED_CAMS[camIndex].eye;
        fovY = FIXED_CAMS[camIndex].fovY;
    }

    Mat4 view = lookAt(eye, target, {0.f, 1.f, 0.f});
    Mat4 proj = perspective(fovY, aspect, 0.1f, 200.f);
    return proj * view;
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
    std::printf("Loaded %zu meshes\n", meshes.size());

    Config cfg;
    const float aspect = static_cast<float>(cfg.width) / cfg.height;

    sf::RenderWindow window(
        sf::VideoMode(cfg.width, cfg.height),
        "Penguin Walk | WASD move | 0-5 camera",
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
    wp.forwardSpeed = 0.f;

    PlayerState player;
    sf::Clock   clock;
    float realTime = 0.f;  // always-advancing simulation time for idle anims
    bool paused   = false;
    int  camIndex = -1;   // -1 = follow, 0-4 = fixed cams 1-5

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed) {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                case sf::Keyboard::Q:     window.close();      break;
                case sf::Keyboard::Space: paused = !paused;    break;
                case sf::Keyboard::R:
                    player = PlayerState{};
                    clock.restart();
                    break;
                // 0 = follow cam, 1-5 = fixed cams
                case sf::Keyboard::Num0: camIndex = -1; break;
                case sf::Keyboard::Num1: camIndex =  0; break;
                case sf::Keyboard::Num2: camIndex =  1; break;
                case sf::Keyboard::Num3: camIndex =  2; break;
                case sf::Keyboard::Num4: camIndex =  3; break;
                case sf::Keyboard::Num5: camIndex =  4; break;
                default: break;
                }
            }
        }

        float dt = std::min(clock.restart().asSeconds(), 0.05f);

        if (!paused) {
            bool wDown = sf::Keyboard::isKeyPressed(sf::Keyboard::W);
            bool sDown = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
            bool aDown = sf::Keyboard::isKeyPressed(sf::Keyboard::A);
            bool dDown = sf::Keyboard::isKeyPressed(sf::Keyboard::D);

            if (aDown) player.yaw += PlayerState::TURN_RATE * dt;
            if (dDown) player.yaw -= PlayerState::TURN_RATE * dt;

            if (wDown && !sDown) {
                player.speed += PlayerState::ACCEL * dt;
                player.speed  = std::min(player.speed, PlayerState::MAX_SPEED);
            } else if (sDown && !wDown) {
                player.speed -= PlayerState::ACCEL * dt;
                player.speed  = std::max(player.speed, -PlayerState::MAX_SPEED);
            } else {
                if (player.speed > 0.f)
                    player.speed = std::max(0.f, player.speed - PlayerState::DECEL * dt);
                else if (player.speed < 0.f)
                    player.speed = std::min(0.f, player.speed + PlayerState::DECEL * dt);
            }

            float fwdX = std::sin(player.yaw);
            float fwdZ = std::cos(player.yaw);
            player.posX += fwdX * player.speed * dt;
            player.posZ += fwdZ * player.speed * dt;

            float speedRatio = std::abs(player.speed) / PlayerState::MAX_SPEED;
            player.animPhase += dt * speedRatio;
            realTime          += dt;

            wp.speedRatio = speedRatio;
            wp.idleTime   = realTime;

            anim.pose(player.animPhase, wp);
        }

        Mat4 vp = computeViewProj(camIndex, player, cfg, aspect);

        Mat4 root = penguinRootTransform(player.posX, player.posZ, player.yaw);
        Animator animWorld = anim;
        for (int j = 0; j < J_COUNT; ++j)
            animWorld.sk.world[j] = root * anim.sk.world[j];

        renderFrame(fb, meshes, animWorld, vp, cfg.lightDir,
                    player.posX, player.posZ, cfg);

        uploadToTexture(fb, frameTex);
        window.clear();
        window.draw(frameSprite);
        if (hasFont) drawHUD(window, font, player, paused, camIndex);
        window.display();
    }

    return 0;
}