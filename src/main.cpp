// main.cpp - Interactive WASD-controlled penguin walk cycle viewer.
//
// Controls:
//   W / S       - walk forward / backward
//   A / D       - turn left / right
//   F           - toggle walk / run mode
//   0           - third-person follow camera (default)
//   1-5         - fixed cameras (Resident Evil style)
//   Space       - pause / resume
//   R           - reset position and orientation
//   Esc / Q     - quit

#include "math_utils.h"
#include "obj_loader.h"
#include "skeleton.h"
#include "rasterizer.h"
#include "joint_rom_ui.h"

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
    bool  running   = false;   // false = walk, true = run

    // Walk limits
    static constexpr float WALK_MAX_SPEED = 2.0f;
    // Run limits (2.5x faster)
    static constexpr float RUN_MAX_SPEED  = 5.0f;

    // Animation phase multiplier for walk vs run
    // Run anim plays at 2.5x the walk cadence so legs/wings move visibly faster.
    static constexpr float WALK_ANIM_RATE = 1.0f;
    static constexpr float RUN_ANIM_RATE  = 2.5f;

    static constexpr float ACCEL     = 4.0f;
    static constexpr float DECEL     = 6.0f;
    static constexpr float TURN_RATE = 2.2f;

    float maxSpeed()   const { return running ? RUN_MAX_SPEED  : WALK_MAX_SPEED; }
    float animRate()   const { return running ? RUN_ANIM_RATE  : WALK_ANIM_RATE; }
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
// ---------------------------------------------------------------------------
// Speedometer widget - drawn top-right corner.
// Shows two horizontal bars: WALK (blue) and RUN (orange), with a needle on
// each indicating the current speed expressed as a fraction of that mode's
// maximum.  When the active mode is highlighted with a bright border.
// ---------------------------------------------------------------------------
static void drawSpeedometer(sf::RenderWindow& win, const sf::Font& font,
                             const PlayerState& ps)
{
    const float WIN_W  = static_cast<float>(win.getSize().x);
    const float BAR_W  = 160.f;
    const float BAR_H  = 18.f;
    const float PAD    = 10.f;
    const float ORIGIN_X = WIN_W - BAR_W - PAD;
    const float ORIGIN_Y = PAD + 30.f;  // below camera label

    // Background panel
    sf::RectangleShape panel(sf::Vector2f(BAR_W + 20.f, 90.f));
    panel.setPosition(ORIGIN_X - 10.f, ORIGIN_Y - 8.f);
    panel.setFillColor(sf::Color(0, 0, 0, 160));
    panel.setOutlineThickness(1.f);
    panel.setOutlineColor(sf::Color(180,180,180,120));
    win.draw(panel);

    // Title
    sf::Text title;
    title.setFont(font);
    title.setString("SPEEDOMETER  [F]");
    title.setCharacterSize(11);
    title.setFillColor(sf::Color(200, 200, 200, 230));
    title.setPosition(ORIGIN_X - 5.f, ORIGIN_Y - 5.f);
    win.draw(title);

    // Helper lambda: draw one labelled speed bar
    auto drawBar = [&](float yOff, const char* label,
                       float curSpeed, float maxSpd,
                       bool active,
                       sf::Color barColor)
    {
        float fy   = ORIGIN_Y + yOff;
        float fill = std::min(1.f, std::abs(curSpeed) / maxSpd);

        // Track background
        sf::RectangleShape track(sf::Vector2f(BAR_W, BAR_H));
        track.setPosition(ORIGIN_X, fy);
        track.setFillColor(sf::Color(40, 40, 40, 200));
        if (active) {
            track.setOutlineThickness(2.f);
            track.setOutlineColor(barColor);
        }
        win.draw(track);

        // Filled portion
        if (fill > 0.f) {
            sf::RectangleShape filled(sf::Vector2f(BAR_W * fill, BAR_H));
            filled.setPosition(ORIGIN_X, fy);
            filled.setFillColor(barColor);
            win.draw(filled);
        }

        // Needle (thin vertical line at fill position)
        float needleX = ORIGIN_X + BAR_W * fill;
        sf::VertexArray needle(sf::Lines, 2);
        needle[0] = sf::Vertex(sf::Vector2f(needleX, fy - 2.f), sf::Color::White);
        needle[1] = sf::Vertex(sf::Vector2f(needleX, fy + BAR_H + 2.f), sf::Color::White);
        win.draw(needle);

        // Label and numeric speed
        char spdbuf[32];
        std::snprintf(spdbuf, sizeof(spdbuf), "%s %.2f/%.2f", label, std::abs(curSpeed), maxSpd);
        sf::Text txt;
        txt.setFont(font);
        txt.setString(spdbuf);
        txt.setCharacterSize(11);
        txt.setFillColor(active ? sf::Color(255,255,255,230) : sf::Color(160,160,160,180));
        txt.setPosition(ORIGIN_X, fy + BAR_H + 2.f);
        win.draw(txt);
    };

    // Walk bar (steel blue)
    drawBar(14.f, "WALK",
            ps.running ? 0.f : ps.speed,
            PlayerState::WALK_MAX_SPEED,
            !ps.running,
            sf::Color(70, 130, 200, 200));

    // Run bar (orange)
    drawBar(54.f, "RUN ",
            ps.running ? ps.speed : 0.f,
            PlayerState::RUN_MAX_SPEED,
            ps.running,
            sf::Color(230, 130, 30, 200));
}

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

    const char* gaitLabel = ps.running ? "RUN" : "WALK";
    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "%s | %s | %s | spd=%.2f | [WASD] move  [F] walk/run  [B] backflip  [0-5] cam  [Space] pause  [R] reset  [Esc] quit",
        paused ? "PAUSED" : "PLAYING", camName, gaitLabel, ps.speed);

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

    // Gait mode label
    sf::Text gaitText;
    gaitText.setFont(font);
    gaitText.setString(ps.running ? ">> RUN <<" : "  WALK  ");
    gaitText.setCharacterSize(15);
    gaitText.setFillColor(ps.running ? sf::Color(255, 140, 0, 240) : sf::Color(100, 180, 255, 240));
    gaitText.setPosition(10.f, 30.f);
    win.draw(gaitText);

    // Speedometer widget
    drawSpeedometer(win, font, ps);
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
    bool showROM  = false; // toggle joint ROM overlay with J
    std::vector<JointROM> jointROMs = computeJointROMs(wp);
    BackflipState backflip;

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
                case sf::Keyboard::J:    showROM = !showROM; break;
                case sf::Keyboard::F:
                    player.running = !player.running;
                    // Clamp speed to new mode's limit when switching
                    if (player.speed > player.maxSpeed())
                        player.speed = player.maxSpeed();
                    else if (player.speed < -player.maxSpeed())
                        player.speed = -player.maxSpeed();
                    break;
                case sf::Keyboard::B:    backflip.trigger(); break;
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
                player.speed  = std::min(player.speed, player.maxSpeed());
            } else if (sDown && !wDown) {
                player.speed -= PlayerState::ACCEL * dt;
                player.speed  = std::max(player.speed, -player.maxSpeed());
            } else {
                if (player.speed > 0.f)
                    player.speed = std::max(0.f, player.speed - PlayerState::DECEL * dt);
                else if (player.speed < 0.f)
                    player.speed = std::min(0.f, player.speed + PlayerState::DECEL * dt);
            }

            float fwdX = std::sin(player.yaw);
            float fwdZ = std::cos(player.yaw);

            // Foot-contact-driven movement: position advances in an impulse at
            // each footstrike rather than gliding at constant speed.
            //
            // phi2 is the double-frequency gait phase (advances once per step,
            // twice per stride).  cos(phi2) oscillates between +1 (footstrike)
            // and -1 (mid-swing), so footDrive = 1 + cos(phi2) peaks at 2 each
            // time a foot hits the ground and drops to 0 between steps.
            // Because the average of (1 + cos) over a full cycle is exactly 1,
            // the average speed per stride equals player.speed unchanged.
            const float PI_X2 = 2.f * 3.14159265f;
            float phi2     = PI_X2 * 2.f * (player.animPhase / wp.period);
            float footDrive = 1.f + std::cos(phi2);
            player.posX += fwdX * player.speed * footDrive * dt;
            player.posZ += fwdZ * player.speed * footDrive * dt;

            float speedRatio = std::abs(player.speed) / player.maxSpeed();
            // Advance animPhase faster when running so joints move visibly quicker.
            player.animPhase += dt * speedRatio * player.animRate();
            realTime          += dt;

            wp.speedRatio = speedRatio;
            wp.idleTime   = realTime;

            // Update backflip animation
            backflip.update(dt);

            // During backflip, stop movement
            if (backflip.isActive()) {
                wp.speedRatio = 0.f;
            }

            anim.pose(player.animPhase, wp);

            // Apply backflip overrides after the walk pose
            if (backflip.isActive()) {
                applyBackflipPose(anim.sk, backflip);
                anim.sk.updateWorld();
            }
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
        if (showROM) {
            drawJointROMArcs(window, vp, cfg.width, cfg.height,
                             animWorld.sk, jointROMs);
            if (hasFont)
                drawROMPanel(window, font, animWorld.sk, jointROMs);
        }
        if (hasFont) drawHUD(window, font, player, paused, camIndex);
        window.display();
    }

    return 0;
}