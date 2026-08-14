def main() -> int:
    print("Run component scripts directly:")
    print("  python src/rgb_to_mesh/colmap.py CONFIG --run-id RUN_ID")
    print("  python src/rgb_to_mesh/metric3d.py --images IMAGES --output OUTPUT")
    print(
        "  python src/rgb_to_mesh/dn_splatter.py --output OUTPUT "
        "--transforms TRANSFORMS --images IMAGES --depth DEPTH"
    )
    print(
        "  python src/rgb_to_mesh/mesh.py --dn-splatter DN_SPLATTER "
        "--output OUTPUT"
    )
    print(
        "  python src/layout_skeleton/oneformer.py --images IMAGES "
        "--mesh-manifest MESH_MANIFEST --model-dir MODEL_DIR --output OUTPUT"
    )
    print(
        "  python src/layout_skeleton/skeleton.py --transforms TRANSFORMS "
        "--dn-splatter DN_SPLATTER --mesh-manifest MESH_MANIFEST "
        "--oneformer ONEFORMER --ns-render NS_RENDER "
        "--superpoint-repo SUPERPOINT_REPO --output OUTPUT"
    )
    print(
        "  python src/prototype_fitting/polygon_init.py --skeleton SKELETON "
        "--output OUTPUT"
    )
    print(
        "  python src/prototype_fitting/prototype.py --skeleton SKELETON "
        "--polygon-init POLYGON_INIT --source-repo SOURCE --output OUTPUT"
    )
    print(
        "  python src/scene_graph/graph.py --prototype PROTOTYPE "
        "--skeleton SKELETON --output OUTPUT"
    )
    print(
        "  python src/layout_export/layout.py --scene-graph SCENE_GRAPH "
        "--prototype PROTOTYPE --output OUTPUT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
