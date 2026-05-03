// joint_rom_ui.h - Joint Range of Motion (ROM) 3D arc visualization and 2D panel.
//
// Draws colored arcs at each animated joint showing the rotation range for
// each active degree of freedom, with a marker at the current angle.
// A side panel shows numeric stats (current angle, min/max in degrees).
//
// Toggle with the 'J' key.

#ifndef JOINT_ROM_UI_H
#define JOINT_ROM_UI_H

#include "math_utils.h"
#include "skeleton.h"

#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-axis ROM specification for a single joint.
// ---------------------------------------------------------------------------
struct AxisROM {
    float minRad{0.f};  // minimum angle in radians
    float maxRad{0.f};  // maximum angle in radians
    bool  active{false};
};

struct JointROM {
    int       jointId;
    std::string label;
    AxisROM   x;  // pitch
    AxisROM   y;  // yaw
    AxisROM   z;  // roll
};

// ---------------------------------------------------------------------------
// Derive ROM limits from WalkParams for each animated joint.
// ---------------------------------------------------------------------------
inline std::vector<JointROM> computeJointROMs(const WalkParams& wp) {
    std::vector<JointROM> roms;

    // PELVIS: roll = [-waddleRoll, +waddleRoll], pitch = [0, leanPitch + small bob]
    roms.push_back({J_PELVIS, "Pelvis",
        {0.f, wp.leanPitch + 0.02f, true},                     // x: lean/pitch
        {0.f, 0.f, false},                                     // y: none
        {-wp.waddleRoll - 0.035f, wp.waddleRoll + 0.035f, true} // z: waddle roll
    });

    // NECK: roll counteracts pelvis waddle (30%)
    roms.push_back({J_NECK, "Neck",
        {0.f, 0.f, false},
        {0.f, 0.f, false},
        {-wp.waddleRoll * 0.30f, wp.waddleRoll * 0.30f, true}
    });

    // HEAD: pitch nod + yaw counter-sway
    roms.push_back({J_HEAD, "Head",
        {-wp.headPitchAmp, wp.headPitchAmp, true},
        {-(wp.headYawAmp + 0.030f), wp.headYawAmp + 0.030f, true},
        {0.f, 0.f, false}
    });

    // BEAK_A (upper): pitch opens up
    roms.push_back({J_BEAK_A, "Beak Upper",
        {0.f, wp.beakOpenAmp * 0.4f, true},
        {0.f, 0.f, false},
        {0.f, 0.f, false}
    });

    // BEAK_B (lower): pitch drops down
    roms.push_back({J_BEAK_B, "Beak Lower",
        {-wp.beakOpenAmp, 0.f, true},
        {0.f, 0.f, false},
        {0.f, 0.f, false}
    });

    // WING_L: pitch swing + roll (flap + hold-out)
    roms.push_back({J_WING_L, "Wing L",
        {-(wp.wingBase + wp.wingSwing + 0.06f), -(wp.wingBase - wp.wingSwing), true},
        {0.f, 0.f, false},
        {-(0.42f + 0.06f + wp.wingFlapAmp), -(0.42f - 0.06f - wp.wingFlapAmp), true}
    });

    // WING_R: pitch swing + roll (flap + hold-out)
    roms.push_back({J_WING_R, "Wing R",
        {-(wp.wingBase + wp.wingSwing + 0.06f), -(wp.wingBase - wp.wingSwing), true},
        {0.f, 0.f, false},
        {0.42f - 0.06f - wp.wingFlapAmp, 0.42f + 0.06f + wp.wingFlapAmp, true}
    });

    // HIP_L: pitch (stride swing)
    float stride = wp.strideAmplitude;
    roms.push_back({J_HIP_L, "Hip L",
        {-stride, stride, true},
        {0.f, 0.f, false},
        {0.f, 0.f, false}
    });

    // HIP_R: pitch (stride swing)
    roms.push_back({J_HIP_R, "Hip R",
        {-stride, stride, true},
        {0.f, 0.f, false},
        {0.f, 0.f, false}
    });

    return roms;
}

// ---------------------------------------------------------------------------
// Draw a 3D arc projected to screen using SFML vertex arrays.
// The arc lies in a plane at the joint's world position and shows [minAngle,
// maxAngle] with a marker at currentAngle.
// ---------------------------------------------------------------------------

// Helper: get the world-space origin of a joint from its world matrix.
inline Vec3 jointWorldPos(const Mat4& worldMat) {
    return {worldMat.m[0][3], worldMat.m[1][3], worldMat.m[2][3]};
}

// Project a 3D point to SFML screen coords; returns false if behind camera.
inline bool project3D(const Mat4& viewProj, const Vec3& p, int width, int height, sf::Vector2f& out) {
    Projected pr = projectPoint(viewProj, p, width, height);
    if (!pr.visible) return false;
    out = sf::Vector2f(pr.sx, pr.sy);
    return true;
}

// Draw an arc in a given plane (defined by two axes) at a joint position.
// axis1 and axis2 define the arc plane (arc sweeps from axis1 toward axis2).
// color: RGB for the arc; markerColor for the current angle indicator.
inline void drawArc3D(sf::RenderWindow& win, const Mat4& viewProj, int fbWidth, int fbHeight, const Vec3& center, const Vec3& axis1, const Vec3& axis2, float radius, float minAngle, float maxAngle, float currentAngle, sf::Color arcColor, sf::Color markerColor)
{
    const int NUM_SEGMENTS = 24;

    // Draw the arc (range) as a line strip
    sf::VertexArray arc(sf::LineStrip);
    for (int i = 0; i <= NUM_SEGMENTS; ++i) {
        float t = static_cast<float>(i) / NUM_SEGMENTS;
        float angle = minAngle + t * (maxAngle - minAngle);
        Vec3 pt = center + axis1 * (radius * std::cos(angle)) + axis2 * (radius * std::sin(angle));
        sf::Vector2f screenPt;
        if (project3D(viewProj, pt, fbWidth, fbHeight, screenPt)) {
            arc.append(sf::Vertex(screenPt, arcColor));
        }
    }
    if (arc.getVertexCount() >= 2)
        win.draw(arc);

    // Draw min/max endpoint ticks as small crosses
    auto drawTick = [&](float angle, sf::Color col) {
        Vec3 pt = center + axis1 * (radius * std::cos(angle)) + axis2 * (radius * std::sin(angle));
        sf::Vector2f sp;
        if (project3D(viewProj, pt, fbWidth, fbHeight, sp)) {
            sf::VertexArray tick(sf::Lines, 4);
            tick[0] = sf::Vertex(sf::Vector2f(sp.x - 3, sp.y), col);
            tick[1] = sf::Vertex(sf::Vector2f(sp.x + 3, sp.y), col);
            tick[2] = sf::Vertex(sf::Vector2f(sp.x, sp.y - 3), col);
            tick[3] = sf::Vertex(sf::Vector2f(sp.x, sp.y + 3), col);
            win.draw(tick);
        }
    };
    drawTick(minAngle, arcColor);
    drawTick(maxAngle, arcColor);

    // Draw current angle marker (filled circle)
    float clamped = std::fmax(minAngle, std::fmin(maxAngle, currentAngle));
    Vec3 markerPt = center + axis1 * (radius * std::cos(clamped)) + axis2 * (radius * std::sin(clamped));
    sf::Vector2f markerScreen;
    if (project3D(viewProj, markerPt, fbWidth, fbHeight, markerScreen)) {
        sf::CircleShape marker(4.f);
        marker.setFillColor(markerColor);
        marker.setOrigin(4.f, 4.f);
        marker.setPosition(markerScreen);
        win.draw(marker);
    }

    // Draw line from center to current angle marker
    sf::Vector2f centerScreen;
    if (project3D(viewProj, center, fbWidth, fbHeight, centerScreen)) {
        sf::VertexArray line(sf::Lines, 2);
        line[0] = sf::Vertex(centerScreen, markerColor);
        if (project3D(viewProj, markerPt, fbWidth, fbHeight, markerScreen)) {
            line[1] = sf::Vertex(markerScreen, markerColor);
            win.draw(line);
        }
    }
}

// ---------------------------------------------------------------------------
// Draw all joint ROM arcs in 3D
// ---------------------------------------------------------------------------
inline void drawJointROMArcs(sf::RenderWindow& win, const Mat4& viewProj, int fbWidth, int fbHeight, const Skeleton& sk, const std::vector<JointROM>& roms)
{
    // Arc radius in world units - small enough to not clutter
    const float ARC_RADIUS = 0.18f;

    // Colors for each axis
    const sf::Color colX(255, 80, 80, 220);    // red = pitch (X)
    const sf::Color colY(80, 255, 80, 220);    // green = yaw (Y)
    const sf::Color colZ(80, 120, 255, 220);   // blue = roll (Z)
    const sf::Color markerCol(255, 255, 0, 255); // yellow marker

    for (const auto& rom : roms) {
        int jid = rom.jointId;
        const Mat4& W = sk.world[jid];
        Vec3 pos = jointWorldPos(W);

        // Extract local axes from the parent's world matrix (or identity for root).
        // We want the arcs to be in the parent's frame since rotations are local.
        Mat4 parentW = Mat4::identity();
        int parentId = sk.joints[jid].parent;
        if (parentId >= 0) parentW = sk.world[parentId];

        // Parent's world axes
        Vec3 parentX = {parentW.m[0][0], parentW.m[1][0], parentW.m[2][0]};
        Vec3 parentY = {parentW.m[0][1], parentW.m[1][1], parentW.m[2][1]};
        Vec3 parentZ = {parentW.m[0][2], parentW.m[1][2], parentW.m[2][2]};
        parentX = parentX.normalized();
        parentY = parentY.normalized();
        parentZ = parentZ.normalized();

        const Vec3& euler = sk.joints[jid].euler;

        // X-axis rotation (pitch): arc in the YZ plane
        if (rom.x.active) {
            drawArc3D(win, viewProj, fbWidth, fbHeight, pos, parentZ, parentY, ARC_RADIUS, rom.x.minRad, rom.x.maxRad, euler.x, colX, markerCol);
        }

        // Y-axis rotation (yaw): arc in the XZ plane
        if (rom.y.active) {
            drawArc3D(win, viewProj, fbWidth, fbHeight, pos, parentX, parentZ, ARC_RADIUS, rom.y.minRad, rom.y.maxRad, euler.y, colY, markerCol);
        }

        // Z-axis rotation (roll): arc in the XY plane
        if (rom.z.active) {
            drawArc3D(win, viewProj, fbWidth, fbHeight, pos, parentX, parentY, ARC_RADIUS, rom.z.minRad, rom.z.maxRad, euler.z, colZ, markerCol);
        }

        // Draw a small dot at the joint center
        sf::Vector2f sp;
        if (project3D(viewProj, pos, fbWidth, fbHeight, sp)) {
            sf::CircleShape dot(3.f);
            dot.setFillColor(sf::Color(255, 255, 255, 200));
            dot.setOrigin(3.f, 3.f);
            dot.setPosition(sp);
            win.draw(dot);
        }
    }
}

// ---------------------------------------------------------------------------
// Draw 2D side panel with joint stats
// ---------------------------------------------------------------------------
inline void drawROMPanel(sf::RenderWindow& win, const sf::Font& font, const Skeleton& sk, const std::vector<JointROM>& roms)
{
    const float PANEL_X = 10.f;
    const float PANEL_Y = 40.f;
    const float LINE_H  = 15.f;
    const float PANEL_W = 260.f;
    const float RAD2DEG = 180.f / 3.14159265f;

    // Count total lines needed
    int lineCount = 1; // title
    for (const auto& rom : roms) {
        lineCount++; // joint name
        if (rom.x.active) lineCount++;
        if (rom.y.active) lineCount++;
        if (rom.z.active) lineCount++;
    }
    lineCount++; // legend line

    float panelH = PANEL_Y + LINE_H * lineCount + 10.f;

    // Background
    sf::RectangleShape bg(sf::Vector2f(PANEL_W, panelH - PANEL_Y + 20.f));
    bg.setPosition(PANEL_X - 5.f, PANEL_Y - 5.f);
    bg.setFillColor(sf::Color(0, 0, 0, 180));
    bg.setOutlineColor(sf::Color(100, 100, 100, 200));
    bg.setOutlineThickness(1.f);
    win.draw(bg);

    float curY = PANEL_Y;

    // Title
    sf::Text title;
    title.setFont(font);
    title.setString("Joint ROM [J to hide]");
    title.setCharacterSize(12);
    title.setFillColor(sf::Color(255, 220, 80, 240));
    title.setPosition(PANEL_X, curY);
    win.draw(title);
    curY += LINE_H + 2.f;

    // Legend
    sf::Text legend;
    legend.setFont(font);
    legend.setString("X=pitch  Y=yaw  Z=roll  (deg)");
    legend.setCharacterSize(10);
    legend.setFillColor(sf::Color(180, 180, 180, 220));
    legend.setPosition(PANEL_X, curY);
    win.draw(legend);
    curY += LINE_H;

    // Per-joint info
    for (const auto& rom : roms) {
        const Vec3& euler = sk.joints[rom.jointId].euler;

        sf::Text jname;
        jname.setFont(font);
        jname.setString(rom.label);
        jname.setCharacterSize(11);
        jname.setFillColor(sf::Color(220, 220, 220, 240));
        jname.setPosition(PANEL_X, curY);
        win.draw(jname);
        curY += LINE_H;

        auto drawAxisLine = [&](const char* axisName, float cur, const AxisROM& a, sf::Color col) {
            if (!a.active) return;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "  %s: %+6.1f  [%+.1f, %+.1f]", axisName, cur * RAD2DEG, a.minRad * RAD2DEG, a.maxRad * RAD2DEG);
            sf::Text line;
            line.setFont(font);
            line.setString(buf);
            line.setCharacterSize(10);
            line.setFillColor(col);
            line.setPosition(PANEL_X + 4.f, curY);
            win.draw(line);

            // Mini progress bar showing current within range
            float barX = PANEL_X + 180.f;
            float barW = 60.f;
            float barH = 8.f;
            float barY2 = curY + 4.f;

            sf::RectangleShape barBg(sf::Vector2f(barW, barH));
            barBg.setPosition(barX, barY2);
            barBg.setFillColor(sf::Color(40, 40, 40, 200));
            barBg.setOutlineColor(col);
            barBg.setOutlineThickness(0.5f);
            win.draw(barBg);

            // Normalized position [0,1]
            float range = a.maxRad - a.minRad;
            float norm = (range > 1e-6f) ? (cur - a.minRad) / range : 0.5f;
            norm = std::fmax(0.f, std::fmin(1.f, norm));

            // Marker
            float markerX = barX + norm * barW;
            sf::RectangleShape markerBar(sf::Vector2f(2.f, barH));
            markerBar.setPosition(markerX - 1.f, barY2);
            markerBar.setFillColor(sf::Color(255, 255, 0, 240));
            win.draw(markerBar);

            curY += LINE_H;
        };

        drawAxisLine("X", euler.x, rom.x, sf::Color(255, 100, 100, 230));
        drawAxisLine("Y", euler.y, rom.y, sf::Color(100, 255, 100, 230));
        drawAxisLine("Z", euler.z, rom.z, sf::Color(100, 140, 255, 230));
    }
}

#endif // JOINT_ROM_UI_H
