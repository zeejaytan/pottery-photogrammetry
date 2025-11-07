#!/usr/bin/env python3
"""
Diagnostic script to analyze duplicate 3D model issues in photogrammetry output.

This script checks:
1. Multiple COLMAP sparse reconstructions
2. Disconnected mesh components
3. Image registration statistics
4. Provides recommendations for fixing issues
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DiagnosticError(Exception):
    """Raised when diagnostic checks fail."""
    pass


def check_colmap_models(work_dir: Path) -> Dict:
    """Check for multiple COLMAP sparse reconstructions."""
    sparse_dir = work_dir / "sparse"

    if not sparse_dir.exists():
        return {
            "status": "ERROR",
            "message": "No sparse directory found",
            "models": []
        }

    # Find all model directories (typically named 0, 1, 2, etc.)
    model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])

    if not model_dirs:
        return {
            "status": "ERROR",
            "message": "No sparse models found",
            "models": []
        }

    models_info = []
    for model_dir in model_dirs:
        # Read cameras.bin, images.bin, points3D.bin to get stats
        cameras_file = model_dir / "cameras.bin"
        images_file = model_dir / "images.bin"
        points_file = model_dir / "points3D.bin"

        model_info = {
            "name": model_dir.name,
            "path": str(model_dir),
            "has_cameras": cameras_file.exists(),
            "has_images": images_file.exists(),
            "has_points": points_file.exists(),
        }

        # Try to get image count using COLMAP
        if images_file.exists():
            try:
                # Use wc to count images (rough estimate from file size)
                model_info["file_size_mb"] = round(images_file.stat().st_size / 1024 / 1024, 2)
            except Exception as e:
                model_info["error"] = str(e)

        models_info.append(model_info)

    status = "WARNING" if len(models_info) > 1 else "OK"
    message = f"Found {len(models_info)} sparse model(s)"

    return {
        "status": status,
        "message": message,
        "models": models_info,
        "model_count": len(models_info)
    }


def check_validation_report(work_dir: Path) -> Dict:
    """Check the validation report for disconnected components."""
    report_file = work_dir / "validation_report.csv"

    if not report_file.exists():
        return {
            "status": "ERROR",
            "message": "No validation_report.csv found",
            "components": []
        }

    components = []
    try:
        with report_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                components.append({
                    "sherd_id": row.get("sherd_id", ""),
                    "vertex_count": int(row.get("vertex_count", 0)),
                    "face_count": int(row.get("face_count", 0)),
                    "surface_area": float(row.get("surface_area", 0)),
                    "status": row.get("status", ""),
                    "file": row.get("file", "")
                })
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Failed to read validation report: {e}",
            "components": []
        }

    # Analyze components
    passed = [c for c in components if c["status"] == "PASS"]
    large_components = [c for c in passed if c["vertex_count"] > 100000]

    status = "OK"
    if len(large_components) > 1:
        status = "WARNING"
        # Check if there are two similar-sized large components (potential duplicates)
        if len(large_components) >= 2:
            sorted_by_size = sorted(large_components, key=lambda x: x["vertex_count"], reverse=True)
            largest = sorted_by_size[0]["vertex_count"]
            second = sorted_by_size[1]["vertex_count"]
            ratio = second / largest if largest > 0 else 0
            if ratio > 0.7:  # If second is >70% of largest, likely duplicate
                status = "ERROR"

    return {
        "status": status,
        "message": f"Found {len(components)} components, {len(passed)} passed validation",
        "components": components,
        "total_components": len(components),
        "passed_components": len(passed),
        "large_components": len(large_components)
    }


def check_mesh_files(work_dir: Path) -> Dict:
    """Check for mesh files and their sizes."""
    mesh_files = {
        "dense_mesh": work_dir / "scene_dense_mesh.ply",
        "refined_mesh": work_dir / "scene_dense_mesh_refine.ply",
    }

    found_meshes = {}
    for name, path in mesh_files.items():
        if path.exists():
            size_mb = round(path.stat().st_size / 1024 / 1024, 2)
            found_meshes[name] = {
                "path": str(path),
                "size_mb": size_mb,
                "exists": True
            }
        else:
            found_meshes[name] = {
                "path": str(path),
                "exists": False
            }

    # Check for exported sherds
    sherds = list(work_dir.glob("sherd_*.ply"))
    found_meshes["sherds"] = {
        "count": len(sherds),
        "files": [str(s.name) for s in sorted(sherds)]
    }

    return {
        "status": "OK" if found_meshes["refined_mesh"]["exists"] else "ERROR",
        "message": f"Found {len(sherds)} exported sherds",
        "meshes": found_meshes
    }


def analyze_colmap_logs(work_dir: Path) -> Dict:
    """Analyze COLMAP logs for registration statistics."""
    log_dir = work_dir.parent / "pipeline" / "logs"

    if not log_dir.exists():
        return {
            "status": "WARNING",
            "message": "No log directory found",
            "stats": {}
        }

    # Look for recent COLMAP logs
    colmap_logs = list(log_dir.glob("*colmap*.log"))

    if not colmap_logs:
        return {
            "status": "WARNING",
            "message": "No COLMAP logs found",
            "stats": {}
        }

    # Parse the most recent log
    latest_log = sorted(colmap_logs, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    stats = {
        "log_file": str(latest_log.name),
        "features_extracted": None,
        "images_matched": None,
        "registered_images": None,
        "total_points": None
    }

    try:
        with latest_log.open("r") as f:
            content = f.read()
            # Look for key statistics in log
            if "Registered images" in content:
                # Parse registration info
                for line in content.split("\n"):
                    if "registered" in line.lower():
                        # Extract numbers from line
                        pass  # This would need actual COLMAP log parsing
    except Exception as e:
        stats["error"] = str(e)

    return {
        "status": "OK",
        "message": "Log analysis incomplete (manual review recommended)",
        "stats": stats
    }


def generate_recommendations(results: Dict) -> List[str]:
    """Generate recommendations based on diagnostic results."""
    recommendations = []

    # Check COLMAP models
    colmap = results.get("colmap_models", {})
    if colmap.get("model_count", 0) > 1:
        recommendations.append(
            "⚠️  MULTIPLE COLMAP MODELS DETECTED\n"
            f"   Found {colmap['model_count']} separate reconstructions in sparse/ directory.\n"
            "   This means COLMAP couldn't register all images into one model.\n\n"
            "   Fixes:\n"
            "   1. Ensure guided_matching: 1 in config (enables cross-ring connections)\n"
            "   2. Lower init_min_tri_angle to 2.0 (currently 4.0)\n"
            "   3. Lower abs_pose_min_num_inliers to 15 (currently 20)\n"
            "   4. Increase max_num_features to 16000 (currently 12000)\n"
            "   5. Check image overlap between photo groups"
        )

    # Check validation report
    validation = results.get("validation", {})
    large_components = validation.get("large_components", 0)
    if large_components > 1:
        components = validation.get("components", [])
        passed = [c for c in components if c["status"] == "PASS"]
        if len(passed) >= 2:
            sorted_by_size = sorted(passed, key=lambda x: x["vertex_count"], reverse=True)
            recommendations.append(
                "⚠️  MULTIPLE LARGE MESH COMPONENTS DETECTED\n"
                f"   Found {large_components} large components:\n" +
                "\n".join([
                    f"   - {c['sherd_id']}: {c['vertex_count']:,} vertices, {c['surface_area']:.2f} area"
                    for c in sorted_by_size[:5]
                ]) + "\n\n"
                "   This indicates disconnected mesh regions (gaps in reconstruction).\n\n"
                "   Possible causes:\n"
                "   1. Some images failed to register in COLMAP\n"
                "   2. Insufficient overlap between photo groups\n"
                "   3. Dense reconstruction has missing data\n\n"
                "   Fixes:\n"
                "   1. Check COLMAP registration rate (should be >95%)\n"
                "   2. Increase number_views to 8 (currently 6) in OpenMVS densify\n"
                "   3. Retake photos with more overlap between groups"
            )

    # Check if both issues exist
    if colmap.get("model_count", 0) > 1 and large_components > 1:
        recommendations.append(
            "🔴 CRITICAL: Both COLMAP and mesh have multiple models/components\n"
            "   This suggests severe registration failure.\n\n"
            "   Priority actions:\n"
            "   1. Run: diagnose_duplicates.py --verbose to see detailed logs\n"
            "   2. Check photo quality and overlap\n"
            "   3. Adjust COLMAP settings more aggressively\n"
            "   4. Consider using sequential matcher for turntable captures"
        )

    if not recommendations:
        recommendations.append(
            "✅ No obvious issues detected!\n"
            "   - Single COLMAP model found\n"
            "   - Reasonable number of mesh components\n\n"
            "   If you're still seeing duplicates, check:\n"
            "   1. The actual 3D viewer - might be showing the same model twice\n"
            "   2. Whether you're viewing sherd_001.ply AND sherd_002.ply simultaneously\n"
            "   3. Mesh topology with MeshLab or CloudCompare"
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose duplicate 3D model issues in photogrammetry output"
    )
    parser.add_argument(
        "work_dir",
        type=str,
        help="Path to work_colmap_openmvs directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()

    if not work_dir.exists():
        print(f"❌ Error: Directory not found: {work_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Diagnosing: {work_dir}\n")

    # Run diagnostic checks
    results = {
        "work_dir": str(work_dir),
        "colmap_models": check_colmap_models(work_dir),
        "validation": check_validation_report(work_dir),
        "mesh_files": check_mesh_files(work_dir),
        "logs": analyze_colmap_logs(work_dir)
    }

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0)

    # Print results
    print("=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    print()

    print(f"📊 COLMAP Models: [{results['colmap_models']['status']}]")
    print(f"   {results['colmap_models']['message']}")
    if args.verbose and results['colmap_models']['models']:
        for model in results['colmap_models']['models']:
            print(f"   - Model {model['name']}: {model.get('file_size_mb', 'N/A')} MB")
    print()

    print(f"📊 Validation Report: [{results['validation']['status']}]")
    print(f"   {results['validation']['message']}")
    if args.verbose and results['validation']['components']:
        for comp in results['validation']['components'][:10]:
            print(f"   - {comp['sherd_id']}: {comp['vertex_count']:,} vertices ({comp['status']})")
    print()

    print(f"📊 Mesh Files: [{results['mesh_files']['status']}]")
    print(f"   {results['mesh_files']['message']}")
    if args.verbose:
        meshes = results['mesh_files']['meshes']
        for name, info in meshes.items():
            if name != "sherds" and info.get("exists"):
                print(f"   - {name}: {info['size_mb']} MB")
    print()

    print(f"📊 Log Analysis: [{results['logs']['status']}]")
    print(f"   {results['logs']['message']}")
    print()

    # Generate and display recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    recommendations = generate_recommendations(results)
    for i, rec in enumerate(recommendations, 1):
        print(rec)
        if i < len(recommendations):
            print()
            print("-" * 70)
            print()

    # Exit with error code if issues found
    has_errors = any(
        r.get("status") == "ERROR"
        for r in results.values()
        if isinstance(r, dict)
    )
    has_warnings = any(
        r.get("status") == "WARNING"
        for r in results.values()
        if isinstance(r, dict)
    )

    if has_errors:
        sys.exit(2)
    elif has_warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
