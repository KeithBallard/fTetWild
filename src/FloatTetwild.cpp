// This file is part of fTetWild, a software for generating tetrahedral meshes.
//
// Copyright (C) 2019 Yixin Hu <yixin.hu@nyu.edu>
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
//

#include "FloatTetwild.h"
#include <floattetwild/AABBWrapper.h>
#include <floattetwild/FloatTetDelaunay.h>
#include <floattetwild/LocalOperations.h>
#include <floattetwild/MeshImprovement.h>
#include <floattetwild/Parameters.h>
#include <floattetwild/Simplification.h>
#include <floattetwild/Statistics.h>
#include <floattetwild/TriangleInsertion.h>
#include <geogram/mesh/mesh_reorder.h>
#include <geogram/mesh/mesh_repair.h>
#include <igl/Timer.h>
#include <igl/winding_number.h>
#include <bitset>
#include <floattetwild/Logger.hpp>
#include <floattetwild/MeshIO.hpp>
#include <floattetwild/Predicates.hpp>
#include <floattetwild/Types.hpp>
#include <fstream>

namespace floatTetWild {

int tetrahedralization_kernel(GEO::Mesh&        sf_mesh,        // in (modified)
                              std::vector<int>& input_tags,     // in
                              Parameters        params,         // in
                              int               boolean_op,     // in
                              bool              skip_simplify,  // in
                              Mesh&             mesh)                       // out
{
    // While reordering the surface mesh, track the ordering to apply to input_tags
    if (input_tags.size() == sf_mesh.facets.nb()) {
        GEO::Attribute<int> bflags(sf_mesh.facets.attributes(), "bbflags");
        for (int index = 0; index < (int)sf_mesh.facets.nb(); ++index) {
            bflags[index] = input_tags[index];
        }
    }

    if (!sf_mesh.facets.are_simplices()) {
        GEO::mesh_repair(
          sf_mesh, GEO::MeshRepairMode(GEO::MESH_REPAIR_TRIANGULATE | GEO::MESH_REPAIR_QUIET));
    }
    GEO::mesh_reorder(sf_mesh, GEO::MESH_ORDER_MORTON);

    // Apply same reorder permutation to input tags
    if (input_tags.size() == sf_mesh.facets.nb()) {
        GEO::Attribute<int> bflags(sf_mesh.facets.attributes(), "bbflags");
        for (int index = 0; index < (int)sf_mesh.facets.nb(); ++index) {
            input_tags[index] = bflags[index];
        }
    }

    std::vector<Vector3>  input_vertices(sf_mesh.vertices.nb());
    std::vector<Vector3i> input_faces(sf_mesh.facets.nb());
    for (size_t i = 0; i < input_vertices.size(); i++) {
        input_vertices[i] << (sf_mesh.vertices.point(i))[0], (sf_mesh.vertices.point(i))[1],
          (sf_mesh.vertices.point(i))[2];
    }
    for (size_t i = 0; i < input_faces.size(); i++) {
        input_faces[i] << sf_mesh.facets.vertex(i, 0), sf_mesh.facets.vertex(i, 1),
          sf_mesh.facets.vertex(i, 2);
    }

    if (input_vertices.empty() || input_faces.empty()) {
        return EXIT_FAILURE;
    }

    AABBWrapper tree(sf_mesh);
#ifdef NEW_ENVELOPE
    tree.init_sf_tree(input_vertices, input_faces, params.eps);
#endif

    if (!params.init(tree.get_sf_diag())) {
        return EXIT_FAILURE;
    }

    stats().record(StateInfo::init_id, 0, input_vertices.size(), input_faces.size(), -1, -1);

    /////////////////////////////////////////////////
    // STEP 1: Preprocessing (mesh simplification) //
    /////////////////////////////////////////////////

    mesh.params = params;

    igl::Timer timer;

    timer.start();
    simplify(input_vertices, input_faces, input_tags, tree, mesh.params, skip_simplify);
    tree.init_b_mesh_and_tree(input_vertices, input_faces, mesh);
    logger().info("preprocessing {}s", timer.getElapsedTimeInSec());
    logger().info("");
    stats().record(StateInfo::preprocessing_id,
                   timer.getElapsedTimeInSec(),
                   input_vertices.size(),
                   input_faces.size(),
                   -1,
                   -1);
    if (mesh.params.log_level <= 1) {
        output_component(input_vertices, input_faces, input_tags);
    }

    ///////////////////////////////////////
    // STEP 2: Volume tetrahedralization //
    ///////////////////////////////////////

    timer.start();
    std::vector<bool> is_face_inserted(input_faces.size(), false);

    FloatTetDelaunay::tetrahedralize(input_vertices, input_faces, tree, mesh, is_face_inserted);

    logger().info("#v = {}", mesh.get_v_num());
    logger().info("#t = {}", mesh.get_t_num());
    logger().info("tetrahedralizing {}s", timer.getElapsedTimeInSec());
    logger().info("");
    stats().record(StateInfo::tetrahedralization_id,
                   timer.getElapsedTimeInSec(),
                   mesh.get_v_num(),
                   mesh.get_t_num(),
                   -1,
                   -1);

    /////////////////////
    // STEP 3: Cutting //
    /////////////////////

    timer.start();
    insert_triangles(input_vertices, input_faces, input_tags, mesh, is_face_inserted, tree, false);
    logger().info("cutting {}s", timer.getElapsedTimeInSec());
    logger().info("");
    stats().record(StateInfo::cutting_id,
                   timer.getElapsedTimeInSec(),
                   mesh.get_v_num(),
                   mesh.get_t_num(),
                   mesh.get_max_energy(),
                   mesh.get_avg_energy(),
                   std::count(is_face_inserted.begin(), is_face_inserted.end(), false));

    //////////////////////////////////////
    // STEP 4: Volume mesh optimization //
    //////////////////////////////////////

    timer.start();
    optimization(
      input_vertices, input_faces, input_tags, is_face_inserted, mesh, tree, {{1, 1, 1, 1}});
    logger().info("mesh optimization {}s", timer.getElapsedTimeInSec());
    logger().info("");
    stats().record(StateInfo::optimization_id,
                   timer.getElapsedTimeInSec(),
                   mesh.get_v_num(),
                   mesh.get_t_num(),
                   mesh.get_max_energy(),
                   mesh.get_avg_energy());

    /////////////////////////////////
    // STEP 5: Interior extraction //
    /////////////////////////////////

    timer.start();
    correct_tracked_surface_orientation(mesh, tree);
    logger().info("correct_tracked_surface_orientation done");
    if (boolean_op < 0) {
        //        filter_outside(mesh);
        if (params.smooth_open_boundary) {
            smooth_open_boundary(mesh, tree);
            for (auto& t : mesh.tets) {
                if (t.is_outside)
                    t.is_removed = true;
            }
        }
        else {
            if (!params.disable_filtering) {
                if (params.use_floodfill) {
                    filter_outside_floodfill(mesh);
                }
                else if (params.use_input_for_wn) {
                    filter_outside(mesh, input_vertices, input_faces);
                }
                else
                    filter_outside(mesh);
            }
        }
    }
    else {
        boolean_operation(mesh, boolean_op);
    }
    stats().record(StateInfo::wn_id,
                   timer.getElapsedTimeInSec(),
                   mesh.get_v_num(),
                   mesh.get_t_num(),
                   mesh.get_max_energy(),
                   mesh.get_avg_energy());
    logger().info("after winding number");
    logger().info("#v = {}", mesh.get_v_num());
    logger().info("#t = {}", mesh.get_t_num());
    logger().info("winding number {}s", timer.getElapsedTimeInSec());
    logger().info("");

    if (!params.log_path.empty()) {
        std::ofstream fout(params.log_path + "_" + params.postfix + ".csv");
        if (fout) {
            fout << stats();
        }
    }

    if (!params.envelope_log.empty()) {
        std::ofstream fout(params.envelope_log);
        fout << envelope_log_csv;
    }

    return 0;
}

int tetrahedralization(GEO::Mesh&       sf_mesh,
                       Parameters       params,
                       Eigen::MatrixXd& VO,
                       Eigen::MatrixXi& TO,
                       int              boolean_op,
                       bool             skip_simplify)
{
    Mesh             mesh;
    std::vector<int> input_tags(sf_mesh.facets.nb(), 0);
    int              return_code =
      tetrahedralization_kernel(sf_mesh, input_tags, params, boolean_op, skip_simplify, mesh);

    if (return_code != 0)
        return return_code;

    MeshIO::extract_volume_mesh(mesh, VO, TO, false);

    return 0;
}

int tetrahedralization(const Eigen::MatrixXd&               all_vertices,
                       const std::vector<Eigen::MatrixX3i>& triangles_by_surface,
                       Parameters                           params,
                       Eigen::MatrixXd&                     volume_vertices,
                       Eigen::MatrixXi&                     volume_tetrahedra,
                       std::vector<unsigned long long>&     volume_attributes,
                       const bool                           skip_simplify)
{
    GEO::initialize();

    GEO::Mesh sf_mesh;

    auto n_surfaces = triangles_by_surface.size();

    Eigen::Index n_vertices = all_vertices.rows();

    int v = 0;
    sf_mesh.vertices.create_vertices(n_vertices);
    for (int i = 0; i < all_vertices.rows(); ++i) {
        GEO::vec3& p = sf_mesh.vertices.point(v++);
        p[0]         = all_vertices(i, 0);
        p[1]         = all_vertices(i, 1);
        p[2]         = all_vertices(i, 2);
    }

    Eigen::Index n_triangles = 0;
    for (auto& triangles : triangles_by_surface) {
        n_triangles += triangles.rows();
    }

    std::vector<int> triangle_tags(n_triangles, 0);
    int              t = 0;
    sf_mesh.facets.create_triangles(n_triangles);
    for (auto surface_index = 0; surface_index < n_surfaces; surface_index++) {
        for (int i = 0; i < triangles_by_surface[surface_index].rows(); ++i) {
            sf_mesh.facets.set_vertex(t, 0, triangles_by_surface[surface_index](i, 0));
            sf_mesh.facets.set_vertex(t, 1, triangles_by_surface[surface_index](i, 1));
            sf_mesh.facets.set_vertex(t, 2, triangles_by_surface[surface_index](i, 2));
            triangle_tags[t] = surface_index;
            t++;
        }
    }

    Mesh mesh;
    int  return_code =
      tetrahedralization_kernel(sf_mesh, triangle_tags, params, -1, skip_simplify, mesh);

    if (return_code != 0)
        return return_code;

    MeshIO::extract_volume_mesh(mesh, volume_vertices, volume_tetrahedra, false);

    assert(n_surfaces < std::numeric_limits<unsigned long long>::digits);
    using RegionBitset = std::bitset<std::numeric_limits<unsigned long long>::digits>;

    Eigen::MatrixXd centroids(mesh.get_t_num(), 3);
    centroids.setZero();
    int index = 0;
    for (size_t i = 0; i < mesh.tets.size(); i++) {
        if (mesh.tets[i].is_removed)
            continue;
        for (int j = 0; j < 4; j++)
            centroids.row(index) += mesh.tet_vertices[mesh.tets[i][j]].pos.cast<double>();
        centroids.row(index) /= 4.0;
        index++;
    }

    std::vector<RegionBitset> tet_tag_bitsets(index);
    for (auto surface_index = 0; surface_index < n_surfaces; surface_index++) {
        Eigen::VectorXd wn;
        igl::winding_number(all_vertices, triangles_by_surface[surface_index], centroids, wn);

        for (auto i = 0; i < wn.size(); i++) {
            if (wn(i) > 0.5) {
                tet_tag_bitsets[i].flip(surface_index);
            }
        }
    }

    volume_attributes.clear();
    volume_attributes.reserve(mesh.tets.size());
    index = 0;
    for (size_t i = 0; i < mesh.tets.size(); i++) {
        if (mesh.tets[i].is_removed)
            continue;
        volume_attributes.push_back(tet_tag_bitsets[index++].to_ullong());
    }

    return 0;
}

int tetrahedralization(const std::vector<Eigen::MatrixXd>&  vertices_by_surface,
                       const std::vector<Eigen::MatrixX3i>& triangles_by_surface,
                       Parameters                           params,
                       Eigen::MatrixXd&                     volume_vertices,
                       Eigen::MatrixXi&                     volume_tetrahedra,
                       std::vector<unsigned long long>&     volume_attributes,
                       const bool                           skip_simplify)
{
    GEO::initialize();

    GEO::Mesh sf_mesh;

    auto n_surfaces = vertices_by_surface.size();

    Eigen::Index n_vertices = 0;
    for (auto& vertices : vertices_by_surface) {
        n_vertices += vertices.rows();
    }

    int              v            = 0;
    std::vector<int> vert_offsets = {0};
    sf_mesh.vertices.create_vertices(n_vertices);
    for (auto surface_index = 0; surface_index < n_surfaces; surface_index++) {
        for (int i = 0; i < vertices_by_surface[surface_index].rows(); ++i) {
            GEO::vec3& p = sf_mesh.vertices.point(v++);
            p[0]         = vertices_by_surface[surface_index](i, 0);
            p[1]         = vertices_by_surface[surface_index](i, 1);
            p[2]         = vertices_by_surface[surface_index](i, 2);
        }
        vert_offsets.push_back(vert_offsets.back() + vertices_by_surface[surface_index].rows());
    }

    Eigen::Index n_triangles = 0;
    for (auto& triangles : triangles_by_surface) {
        n_triangles += triangles.rows();
    }

    std::vector<int> triangle_tags(n_triangles, 0);
    int              t = 0;
    sf_mesh.facets.create_triangles(n_triangles);
    for (auto surface_index = 0; surface_index < n_surfaces; surface_index++) {
        auto& o = vert_offsets[surface_index];
        for (int i = 0; i < triangles_by_surface[surface_index].rows(); ++i) {
            sf_mesh.facets.set_vertex(t, 0, triangles_by_surface[surface_index](i, 0) + o);
            sf_mesh.facets.set_vertex(t, 1, triangles_by_surface[surface_index](i, 1) + o);
            sf_mesh.facets.set_vertex(t, 2, triangles_by_surface[surface_index](i, 2) + o);
            triangle_tags[t] = surface_index;
            t++;
        }
    }

    Mesh mesh;
    int  return_code =
      tetrahedralization_kernel(sf_mesh, triangle_tags, params, -1, skip_simplify, mesh);

    if (return_code != 0)
        return return_code;

    MeshIO::extract_volume_mesh(mesh, volume_vertices, volume_tetrahedra, false);

    assert(n_surfaces < std::numeric_limits<unsigned long long>::digits);
    using RegionBitset = std::bitset<std::numeric_limits<unsigned long long>::digits>;

    Eigen::MatrixXd centroids(mesh.get_t_num(), 3);
    centroids.setZero();
    int index = 0;
    for (size_t i = 0; i < mesh.tets.size(); i++) {
        if (mesh.tets[i].is_removed)
            continue;
        for (int j = 0; j < 4; j++)
            centroids.row(index) += mesh.tet_vertices[mesh.tets[i][j]].pos.cast<double>();
        centroids.row(index) /= 4.0;
        index++;
    }

    std::vector<RegionBitset> tet_tag_bitsets(index);
    for (auto surface_index = 0; surface_index < n_surfaces; surface_index++) {
        Eigen::VectorXd wn;
        igl::winding_number(
          vertices_by_surface[surface_index], triangles_by_surface[surface_index], centroids, wn);

        for (auto i = 0; i < wn.size(); i++) {
            if (wn(i) > 0.5) {
                tet_tag_bitsets[i].flip(surface_index);
            }
        }
    }

    volume_attributes.clear();
    volume_attributes.reserve(mesh.tets.size());
    index = 0;
    for (size_t i = 0; i < mesh.tets.size(); i++) {
        if (mesh.tets[i].is_removed)
            continue;
        volume_attributes.push_back(tet_tag_bitsets[index++].to_ullong());
    }

    return 0;
}

}  // namespace floatTetWild
