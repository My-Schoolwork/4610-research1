/*
 * Task 3: Cubism Animal via Hierarchical Modelling
 *
 * Build a blocky animal (or robot) from transformed unit cubes arranged
 * in a parent-child tree. Export the result as a Wavefront OBJ file.
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./task3_cubism_animal
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <string>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// ---- Data Structures ----

struct Triangle {
    int v0, v1, v2;   // 0-based vertex indices
};

struct Mesh {
    vector<Vector3f> vertices;
    vector<Triangle> faces;
};

struct ModelNode {
    string name;
    Vector3f position;
    Vector3f rotation;
    Vector3f scale;

    vector<unique_ptr<ModelNode>> children;

    ModelNode(const string& name,
              const Vector3f& pos = Vector3f::Zero(),
              const Vector3f& rot = Vector3f::Zero(),
              const Vector3f& scl = Vector3f::Ones())
        : name(name), position(pos), rotation(rot), scale(scl) {}

    ModelNode* addChild(const string& name,
                        const Vector3f& pos,
                        const Vector3f& scl,
                        const Vector3f& rot = Vector3f::Zero()) {
        children.push_back(make_unique<ModelNode>(name, pos, rot, scl));
        return children.back().get();
    }
};

// ---- Transformation Matrices ----

Matrix4f translationMatrix(const Vector3f& t) {
    Matrix4f M = Matrix4f::Identity();
    M(0, 3) = t.x();
    M(1, 3) = t.y();
    M(2, 3) = t.z();
    return M;
}

Matrix4f scalingMatrix(const Vector3f& s) {
    Matrix4f M = Matrix4f::Identity();
    M(0, 0) = s.x();
    M(1, 1) = s.y();
    M(2, 2) = s.z();
    return M;
}

// rotation around X axis (degrees -> radians)
Matrix4f rotationX(float degrees) {
    float r = degrees * M_PI / 180.0f;
    Matrix4f M = Matrix4f::Identity();
    M(1, 1) =  cos(r);
    M(1, 2) = -sin(r);
    M(2, 1) =  sin(r);
    M(2, 2) =  cos(r);
    return M;
}

Matrix4f rotationY(float degrees) {
    float r = degrees * M_PI / 180.0f;
    Matrix4f M = Matrix4f::Identity();
    M(0, 0) =  cos(r);
    M(0, 2) =  sin(r);
    M(2, 0) = -sin(r);
    M(2, 2) =  cos(r);
    return M;
}

Matrix4f rotationZ(float degrees) {
    float r = degrees * M_PI / 180.0f;
    Matrix4f M = Matrix4f::Identity();
    M(0, 0) =  cos(r);
    M(0, 1) = -sin(r);
    M(1, 0) =  sin(r);
    M(1, 1) =  cos(r);
    return M;
}

Matrix4f rotationMatrix(const Vector3f& eulerDeg) {
    return rotationZ(eulerDeg.z()) * rotationY(eulerDeg.y()) * rotationX(eulerDeg.x());
}

// ---- Unit Cube ----
Mesh createUnitCube() {
    Mesh mesh;

    // 8 corners of a unit cube centered at origin (each coord is +/-0.5)
    mesh.vertices = {
        Vector3f(-0.5f, -0.5f, -0.5f), // 0: left  bottom back
        Vector3f( 0.5f, -0.5f, -0.5f), // 1: right bottom back
        Vector3f( 0.5f,  0.5f, -0.5f), // 2: right top    back
        Vector3f(-0.5f,  0.5f, -0.5f), // 3: left  top    back
        Vector3f(-0.5f, -0.5f,  0.5f), // 4: left  bottom front
        Vector3f( 0.5f, -0.5f,  0.5f), // 5: right bottom front
        Vector3f( 0.5f,  0.5f,  0.5f), // 6: right top    front
        Vector3f(-0.5f,  0.5f,  0.5f), // 7: left  top    front
    };

    // 12 triangles (2 per face), CCW winding from outside
    mesh.faces = {
        // back face 
        {0, 2, 1}, {0, 3, 2},
        // front face 
        {4, 5, 6}, {4, 6, 7},
        // left face 
        {0, 4, 7}, {0, 7, 3},
        // right face 
        {1, 2, 6}, {1, 6, 5},
        // bottom face 
        {0, 1, 5}, {0, 5, 4},
        // top face y
        {3, 7, 6}, {3, 6, 2},
    };

    return mesh;
}

// ---- Hierarchical Mesh Collection ----
// Important: pass jointWorld (T*R only, no scale) to children, not meshWorld.
void collectMeshes(const ModelNode* node,
                   const Matrix4f& parentJointWorld,
                   vector<Vector3f>& outVerts,
                   vector<Triangle>& outFaces) {

    // Local TRS for this node
    Matrix4f T = translationMatrix(node->position);
    Matrix4f R = rotationMatrix(node->rotation);
    Matrix4f S = scalingMatrix(node->scale);

    // Full mesh transform includes scale
    Matrix4f jointWorld = parentJointWorld * T * R;
    Matrix4f meshWorld = jointWorld * S;

    // Transform unit cube and append to global list
    Mesh cube = createUnitCube();
    int vertOffset = (int)outVerts.size();

    for (const auto& v : cube.vertices) {
        Vector4f vh(v.x(), v.y(), v.z(), 1.0f);
        Vector4f vt = meshWorld * vh;
        outVerts.push_back(Vector3f(vt.x(), vt.y(), vt.z()));
    }

    for (const auto& f : cube.faces) {
        outFaces.push_back({f.v0 + vertOffset, f.v1 + vertOffset, f.v2 + vertOffset});
    }

    // Recurse into children using joint transform (not mesh transform)
    for (const auto& child : node->children) {
        collectMeshes(child.get(), jointWorld, outVerts, outFaces);
    }
}

// ---- OBJ Export ----

void exportOBJ(const string& filename,
               const vector<Vector3f>& vertices,
               const vector<Triangle>& faces) {
    
    std::filesystem::create_directory("./model");

    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "Error: cannot open " << filename << " for writing." << endl;
        return;
    }

    ofs << "# Cubism Animal - Task 3\n";
    ofs << "# Vertices: " << vertices.size() << "\n";
    ofs << "# Faces: " << faces.size() << "\n\n";

    for (const auto& v : vertices) {
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    ofs << "\n";
    // Reminder OBJ indices are 1-based!
    for (const auto& f : faces) {
        ofs << "f " << (f.v0 + 1) << " " << (f.v1 + 1) << " " << (f.v2 + 1) << "\n";
    }

    ofs.close();
    cout << "Exported: " << filename << endl;
}

// ---- Print Tree ----

void printHierarchy(const ModelNode* node, int depth = 0) {
    string indent(depth * 2, ' ');
    cout << indent << "|- " << node->name
         << "  pos(" << node->position.transpose() << ")"
         << "  rot(" << node->rotation.transpose() << ")"
         << "  scale(" << node->scale.transpose() << ")" << endl;
    for (const auto& child : node->children) {
        printHierarchy(child.get(), depth + 1);
    }
}

// ---- Build Animal: Blocky Penguin ----
unique_ptr<ModelNode> buildAnimalModel() {
 
    // body: wider than tall, flat-ish from front to back
    auto body = make_unique<ModelNode>(
        "Body",
        Vector3f(0.0f, 0.0f, 0.0f),
        Vector3f(0.0f, 0.0f, 0.0f),
        Vector3f(1.0f, 1.3f, 0.85f));
 
    // belly patch —
    body->addChild("Belly",
        Vector3f(0.0f, -0.05f, 0.47f),
        Vector3f(0.55f, 0.85f, 0.08f));
    // neck
    auto neck = body->addChild("Neck",
        Vector3f(0.0f, 0.68f, 0.1f),
        Vector3f(0.38f, 0.28f, 0.38f));
    // head
    auto head = neck->addChild("Head",
        Vector3f(0.0f, 0.58f, 0.0f),
        Vector3f(0.72f, 0.72f, 0.68f));
    // beak upper half
    head->addChild("Beak Upper",
        Vector3f(0.0f, -0.18f, 0.44f),
        Vector3f(0.22f, 0.1f, 0.26f),
        Vector3f(-8.0f, 0.0f, 0.0f));
 
    // beak lower half where it droops a bit more
    head->addChild("Beak Lower",
        Vector3f(0.0f, -0.28f, 0.42f),
        Vector3f(0.18f, 0.08f, 0.22f),
        Vector3f(12.0f, 0.0f, 0.0f));
 
    // eyes sit on upper-front face of head
    head->addChild("Left Eye",
        Vector3f(-0.2f, 0.12f, 0.38f),
        Vector3f(0.13f, 0.13f, 0.07f));
 
    head->addChild("Right Eye",
        Vector3f( 0.2f, 0.12f, 0.38f),
        Vector3f(0.13f, 0.13f, 0.07f));
 
    // wings: flat slabs, angled down and outward
    // left wing tilted so tip points down (-Z rotation tilts it)
    body->addChild("Left Wing",
        Vector3f(-0.62f, 0.05f, 0.0f),
        Vector3f(0.18f, 0.65f, 0.28f),
        Vector3f(0.0f, 0.0f, 18.0f));   // lean outward slightly
 
    body->addChild("Right Wing",
        Vector3f( 0.62f, 0.05f, 0.0f),
        Vector3f(0.18f, 0.65f, 0.28f),
        Vector3f(0.0f, 0.0f, -18.0f));
 
    // feet: wide flat blocks below body, splayed forward a bit
    body->addChild("Left Foot",
        Vector3f(-0.28f, -0.78f, 0.22f),
        Vector3f(0.35f, 0.13f, 0.55f));
 
    body->addChild("Right Foot",
        Vector3f( 0.28f, -0.78f, 0.22f),
        Vector3f(0.35f, 0.13f, 0.55f));
 
    return body;
}

// ---- Main ----

int main() {
    cout << "========================================" << endl;
    cout << " Task 3: Cubism Animal" << endl;
    cout << " Hierarchical Modelling" << endl;
    cout << "========================================" << endl << endl;

    auto animal = buildAnimalModel();

    cout << "--- Model Hierarchy ---" << endl;
    printHierarchy(animal.get());
    cout << endl;

    vector<Vector3f> allVertices;
    vector<Triangle> allFaces;
    collectMeshes(animal.get(), Matrix4f::Identity(), allVertices, allFaces);

    exportOBJ("model/cubism_animal.obj", allVertices, allFaces);

    int V = (int)allVertices.size();
    int F = (int)allFaces.size();
    int E = F * 3 / 2;

    cout << endl;
    cout << "--- Model Statistics ---" << endl;
    cout << "Total vertices  (V): " << V << endl;
    cout << "Total edges     (E): " << E << endl;
    cout << "Total faces     (F): " << F << endl;
    cout << "Euler: V - E + F = " << (V - E + F) << endl;

    cout << endl << "Done. Open model/cubism_animal.obj in MeshLab to visualize." << endl;
    return 0;
}