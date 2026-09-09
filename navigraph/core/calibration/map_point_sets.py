"""Reusable sets of map (schema) points for camera-to-map calibration.

The map side of a calibration is the same picture every session, so its
points are the same every session too. A saved point set records them once
and lets a later calibration reuse them: only the camera frame still needs
clicking, and every session then shares one map-side reference instead of a
freshly clicked approximation of it.

Sets live as JSON files (one set per file, named by the file stem) and can
also be declared inline in a config under ``calibrator_parameters``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

from loguru import logger

from .point_selector import Point


#: A set with fewer pairs than this cannot produce a homography.
MIN_POINTS_PER_SET = 4

#: Default location of point-set files, relative to the config directory.
DEFAULT_POINT_SETS_DIR = "./resources/map_point_sets"


@dataclass(frozen=True)
class MapPointSet:
    """A named, ordered set of points on the map image."""

    name: str
    points: Tuple[Point, ...]
    description: str = ""
    map_image: Optional[str] = None
    map_size: Optional[Tuple[int, int]] = None  # (width, height) when recorded
    labels: Tuple[str, ...] = ()
    source_path: Optional[Path] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Map point set requires a name")
        if len(self.points) < MIN_POINTS_PER_SET:
            raise ValueError(
                f"Map point set '{self.name}' has {len(self.points)} points; "
                f"at least {MIN_POINTS_PER_SET} are required"
            )
        if self.labels and len(self.labels) != len(self.points):
            raise ValueError(
                f"Map point set '{self.name}' has {len(self.labels)} labels "
                f"for {len(self.points)} points"
            )

    def label_for(self, index: int) -> str:
        """Human-readable label for a point, falling back to its number."""
        if self.labels:
            return self.labels[index]
        return str(index + 1)

    def size_mismatch(self, map_shape: Sequence[int]) -> Optional[str]:
        """Describe how the recorded map size differs from ``map_shape``.

        Coordinates are stored in pixels, so a map image of a different size
        makes them point somewhere else. Returns None when the sizes match or
        no size was recorded.

        Args:
            map_shape: Shape of the map image as returned by OpenCV (h, w, ...)

        Returns:
            Description of the mismatch, or None if there is nothing to report
        """
        if not self.map_size:
            return None
        height, width = int(map_shape[0]), int(map_shape[1])
        recorded_w, recorded_h = self.map_size
        if (recorded_w, recorded_h) == (width, height):
            return None
        return (
            f"Point set '{self.name}' was recorded on a {recorded_w}x{recorded_h} "
            f"map but this map is {width}x{height}; the points will not line up."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the on-disk JSON structure."""
        data: Dict[str, Any] = {
            "name": self.name,
            "points": [[float(p.x), float(p.y)] for p in self.points],
        }
        if self.description:
            data["description"] = self.description
        if self.map_image:
            data["map_image"] = self.map_image
        if self.map_size:
            data["map_size"] = [int(self.map_size[0]), int(self.map_size[1])]
        if self.labels:
            data["labels"] = list(self.labels)
        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        name: Optional[str] = None,
        source_path: Optional[Path] = None
    ) -> "MapPointSet":
        """Build a set from a dict, as loaded from JSON or a config section.

        Args:
            data: Mapping with at least a "points" entry
            name: Name to use when the data itself doesn't carry one
            source_path: File the data came from, for error messages

        Returns:
            The parsed MapPointSet

        Raises:
            ValueError: If the structure or the point coordinates are invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Map point set must be a mapping, got {type(data).__name__}")

        set_name = data.get("name") or name
        if not set_name:
            raise ValueError("Map point set is missing a name")

        raw_points = data.get("points")
        if raw_points is None:
            raise ValueError(f"Map point set '{set_name}' has no 'points' entry")

        points = tuple(_parse_point(raw, set_name, i) for i, raw in enumerate(raw_points))

        map_size = data.get("map_size")
        if map_size is not None:
            if len(map_size) != 2:
                raise ValueError(
                    f"Map point set '{set_name}': map_size must be [width, height]"
                )
            map_size = (int(map_size[0]), int(map_size[1]))

        labels = tuple(str(label) for label in data.get("labels", ()))

        return cls(
            name=str(set_name),
            points=points,
            description=str(data.get("description", "")),
            map_image=data.get("map_image"),
            map_size=map_size,
            labels=labels,
            source_path=source_path,
        )

    def save(self, path: Path) -> Path:
        """Write the set to a JSON file.

        Args:
            path: Destination file, or a directory to place "<name>.json" in

        Returns:
            Path actually written
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path = path / f"{self.name}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

        logger.info(f"Map point set '{self.name}' saved to: {path}")
        return path


def _parse_point(raw: Any, set_name: str, index: int) -> Point:
    """Parse one point from either [x, y] or {"x": ..., "y": ...}."""
    if isinstance(raw, dict):
        try:
            return Point(float(raw["x"]), float(raw["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Map point set '{set_name}': point {index + 1} needs numeric 'x' and 'y'"
            ) from exc

    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return Point(float(raw[0]), float(raw[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Map point set '{set_name}': point {index + 1} has non-numeric coordinates"
            ) from exc

    raise ValueError(
        f"Map point set '{set_name}': point {index + 1} must be [x, y] or "
        f"{{'x': x, 'y': y}}, got {raw!r}"
    )


def load_map_point_set(path: Path) -> MapPointSet:
    """Load a single point set from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        The parsed MapPointSet (named after the file stem if the file has no name)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file isn't valid point-set JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Map point set file not found: {path}")

    try:
        with open(path) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Map point set file is not valid JSON: {path} ({exc})") from exc

    return MapPointSet.from_dict(data, name=path.stem, source_path=path)


def load_map_point_sets_from_path(path: Path) -> List[MapPointSet]:
    """Load every point set found at a file or directory path.

    A directory is scanned (non-recursively) for "*.json"; unreadable files are
    logged and skipped so one broken file doesn't hide the rest.

    Args:
        path: JSON file or directory of JSON files

    Returns:
        Point sets sorted by name (empty if the path doesn't exist)
    """
    path = Path(path)
    if not path.exists():
        return []

    if path.is_file():
        return [load_map_point_set(path)]

    sets: List[MapPointSet] = []
    for candidate in sorted(path.glob("*.json")):
        try:
            sets.append(load_map_point_set(candidate))
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(f"Skipping map point set {candidate.name}: {exc}")

    return sorted(sets, key=lambda point_set: point_set.name)


def load_map_point_sets_from_config(config_sets: Any) -> List[MapPointSet]:
    """Parse point sets declared inline in a config.

    Accepts either a mapping of name -> spec or a list of specs that each
    carry their own name.

    Args:
        config_sets: The config value to parse (None yields an empty list)

    Returns:
        Point sets sorted by name

    Raises:
        ValueError: If the structure is neither a mapping nor a list
    """
    if not config_sets:
        return []

    sets: List[MapPointSet] = []
    if isinstance(config_sets, dict):
        entries: Iterable[Tuple[Optional[str], Any]] = config_sets.items()
    elif isinstance(config_sets, list):
        entries = ((None, entry) for entry in config_sets)
    else:
        raise ValueError(
            f"calibrator_parameters.map_point_sets must be a mapping or list, "
            f"got {type(config_sets).__name__}"
        )

    for name, spec in entries:
        # A bare list of coordinates under a name is the shorthand form.
        if isinstance(spec, list):
            spec = {"points": spec}
        sets.append(MapPointSet.from_dict(spec, name=name))

    return sorted(sets, key=lambda point_set: point_set.name)


def find_map_point_set(sets: Sequence[MapPointSet], name: str) -> MapPointSet:
    """Look up a set by name (case-insensitive).

    Args:
        sets: Available point sets
        name: Requested name

    Returns:
        The matching set

    Raises:
        ValueError: If no set matches
    """
    for point_set in sets:
        if point_set.name.lower() == name.lower():
            return point_set

    available = ", ".join(point_set.name for point_set in sets) or "none"
    raise ValueError(f"Unknown map point set '{name}'. Available: {available}")
