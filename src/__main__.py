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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
