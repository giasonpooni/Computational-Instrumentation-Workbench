"""Retained square-tiled-flow contracts; no provider import or flow execution.

Checks the supplied segment/event chain and declared gluing, not a separately
computed trajectory. Content consistency is not independent authentication.
"""
from copy import deepcopy
from fractions import Fraction

from .telemetry import canonical as _canonical_json, digest

ARITHMETIC = {
    "kind": "exact_rational", "coordinate_unit": "unit_square_side",
    "time_parameter": "declared_flow_parameter", "tolerance": "0",
    "derivation": "affine_position_and_rational_boundary_intersection",
    "input_bits": 64, "arithmetic_bits": 256,
    "vertex_policy": "stop_before_any_vertex_continuation",
    "endpoint_policy": "process_single_edge_crossing_at_duration",
    "event_budget_policy": "stop_at_next_boundary_before_omitted_gluing",
}
CLAIM_SCOPE = "exact_rational_translation_flow_prefix_on_declared_square_tiled_surface"
EDGE = {"right": (0, -1), "left": (0, 1), "up": (1, -1), "down": (1, 1)}
INVARIANTS = {"affine_segments_match_direction", "segment_continuity_via_gluing",
              "directed_edge_maps_match_permutations", "positions_in_closed_unit_square",
              "event_times_strictly_increasing", "unfolded_displacement_matches",
              "elapsed_within_requested_duration", "requested_duration_completed"}


def canonical(value):
    try:
        return _canonical_json(value)
    except (TypeError, OverflowError, UnicodeEncodeError) as exc:
        raise ValueError("Translation record must contain finite JSON values") from exc


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("Translation record has unexpected or missing fields")


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _fraction(value, bits=256):
    if type(value) is not str or len(value) > (43 if bits == 64 else 160):
        raise ValueError("Translation rational exceeds its text budget")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid translation rational") from exc
    if str(result) != value or max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > bits:
        raise ValueError("Translation rational must be canonical, reduced and within its bit budget")
    return result


def _pair(value, bits=256):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Translation coordinates require two rational strings")
    return [_fraction(item, bits) for item in value]


def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Translation tile index lies outside its declared surface")


def _inverse(permutation):
    return [permutation.index(i) for i in range(len(permutation))]


def validate_request(request):
    """Native 64-bit rational profile, with CIW's tighter 256-event cap."""
    _keys(request, {"schema", "gluing", "start", "direction", "duration", "max_events"})
    if request["schema"] != "tsde.square-tiled-flow-request.v1" or len(canonical(request)) > 32768:
        raise ValueError("Unsupported or oversized translation request")
    _keys(request["gluing"], {"right", "up"})
    right, up = request["gluing"]["right"], request["gluing"]["up"]
    if type(right) is not list or not 1 <= len(right) <= 32:
        raise ValueError("Translation flow requires 1..32 tiles")
    count = len(right)
    for permutation in (right, up):
        if type(permutation) is not list or len(permutation) != count or any(type(v) is not int for v in permutation) or sorted(permutation) != list(range(count)):
            raise ValueError("Translation gluing must contain two complete permutations")
    reached, pending = {0}, [0]
    while pending:
        tile = pending.pop()
        for neighbor in (right[tile], up[tile]):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if len(reached) != count:
        raise ValueError("Translation surface must be connected")
    _keys(request["start"], {"tile", "position"})
    _index(request["start"]["tile"], count)
    if any(not 0 < v < 1 for v in _pair(request["start"]["position"], 64)):
        raise ValueError("Translation start must be strictly inside its square")
    direction = _pair(request["direction"], 64)
    if not any(direction) or any(abs(v) > 1024 for v in direction):
        raise ValueError("Translation direction must be nonzero and bounded by 1024")
    if not 0 <= _fraction(request["duration"], 64) <= 1024:
        raise ValueError("Translation duration must lie in [0,1024]")
    if type(request["max_events"]) is not int or not 0 <= request["max_events"] <= 256:
        raise ValueError("Workbench translation profile permits 0..256 events")
    return deepcopy(request)


def _topology(request, retained):
    """Check corner classes from edge identifications, without tracing a flow."""
    right, up = request["gluing"]["right"], request["gluing"]["up"]
    count = len(right)
    corner_graph = [set() for _ in range(4 * count)]
    for tile in range(count):
        for a, b in ((4*tile+1, 4*right[tile]), (4*tile+2, 4*right[tile]+3),
                     (4*tile+3, 4*up[tile]), (4*tile+2, 4*up[tile]+1)):
            corner_graph[a].add(b)
            corner_graph[b].add(a)
    seen, vertices, lookup = set(), [], {}
    for first in range(4 * count):
        if first in seen:
            continue
        reached, pending = {first}, [first]
        while pending:
            for neighbor in corner_graph[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        seen.update(reached)
        if len(reached) % 4:
            raise ValueError("Invalid translation-surface cone angle")
        index = len(vertices)
        corners = [list(divmod(corner, 4)) for corner in sorted(reached)]
        vertices.append({"vertex_id": index, "corners": corners,
                         "cone_angle_multiple_of_2pi": len(corners)//4, "singular": len(corners) != 4})
        lookup.update({tuple(corner): index for corner in corners})
    chi = len(vertices) - count
    if chi > 0 or chi % 2:
        raise ValueError("Invalid translation-surface Euler characteristic")
    expected = {"permutations_bijective": True, "connected": True, "tile_count": count, "edge_count": 2*count,
                "vertex_count": len(vertices), "euler_characteristic": chi, "genus": 1-chi//2,
                "right": right, "left": _inverse(right), "up": up, "down": _inverse(up), "vertices": vertices}
    _same(retained, expected, "Translation topology or gluing differs from its declared permutations")
    return expected, lookup


def _outgoing(position, direction):
    return [edge for axis, positive, negative in ((0, "right", "left"), (1, "up", "down"))
            for edge in ([positive] if position[axis] == 1 and direction[axis] > 0 else
                         [negative] if position[axis] == 0 and direction[axis] < 0 else [])]


def validate_result(request, data):
    """Validate the retained chain and stop semantics; never invoke a provider."""
    validate_request(request)
    _keys(data, {"schema", "operation_id", "request", "request_digest", "claim_scope", "arithmetic", "gluing_validation",
                 "status", "elapsed", "remaining", "final_state", "segments", "events", "invariants", "artifact_digest"})
    if len(canonical(data)) > 4 * 1024 * 1024:
        raise ValueError("Translation result exceeds its byte budget")
    if data["schema"] != "tsde.square-tiled-flow-result.v1" or data["operation_id"] != "tsde.square-tiled-flow.v1" or data["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("Unsupported translation result schema, operation or claim scope")
    _same(data["request"], request, "Translation result names another request")
    if data["request_digest"] != digest(request) or data["artifact_digest"] != digest({k:v for k,v in data.items() if k != "artifact_digest"}):
        raise ValueError("Translation content identity differs")
    _same(data["arithmetic"], ARITHMETIC, "Translation arithmetic or stopping policy changed")
    topology, vertices = _topology(request, data["gluing_validation"])
    status = data["status"]
    if status not in ("completed", "stopped_at_vertex", "event_budget_exhausted"):
        raise ValueError("Unsupported translation result status")
    elapsed, remaining, duration = _fraction(data["elapsed"]), _fraction(data["remaining"]), _fraction(request["duration"], 64)
    if elapsed < 0 or remaining < 0 or elapsed + remaining != duration:
        raise ValueError("Translation duration accounting differs")
    segments, events = data["segments"], data["events"]
    if type(segments) is not list or len(segments) > request["max_events"] + 1 or type(events) is not list or len(events) > request["max_events"]:
        raise ValueError("Translation segment or event budget exceeded")
    direction = _pair(request["direction"], 64)
    position, tile, time, event_index = _pair(request["start"]["position"], 64), request["start"]["tile"], Fraction(0), 0
    previous_event = Fraction(-1)
    for event in events:
        _keys(event, {"index", "time", "edge", "from_tile", "to_tile", "from_position", "to_position", "translation"})
        stamp = _fraction(event["time"])
        if not previous_event < stamp <= elapsed:
            raise ValueError("Translation events must have strictly increasing in-range times")
        previous_event = stamp
    for index, segment in enumerate(segments):
        _keys(segment, {"tile", "t_start", "t_end", "start", "end"})
        _index(segment["tile"], topology["tile_count"])
        start, end = _pair(segment["start"]), _pair(segment["end"])
        before, after = _fraction(segment["t_start"]), _fraction(segment["t_end"])
        if segment["tile"] != tile or start != position or before != time or not before < after <= elapsed:
            raise ValueError("Translation segment continuity or time progression differs")
        if any(not 0 <= v <= 1 for v in start + end) or end != [start[i] + direction[i]*(after-before) for i in range(2)]:
            raise ValueError("Translation segment differs from declared affine motion or square bounds")
        time, position = after, end
        if event_index < len(events) and _fraction(events[event_index]["time"]) == time:
            event = events[event_index]
            edge = event["edge"]
            if type(edge) is not str or edge not in EDGE or _outgoing(position, direction) != [edge]:
                raise ValueError("Translation event must be a single outgoing edge, never a corner continuation")
            axis, amount = EDGE[edge]
            translation = [Fraction(0), Fraction(0)]
            translation[axis] = Fraction(amount)
            target = [position[i] + translation[i] for i in range(2)]
            expected = {"index": event_index, "time": str(time), "edge": edge, "from_tile": tile,
                        "to_tile": topology[edge][tile], "from_position": list(map(str, position)),
                        "to_position": list(map(str, target)), "translation": list(map(str, translation))}
            _same(event, expected, "Translation event map, index or position differs from its gluing")
            tile, position = expected["to_tile"], target
            event_index += 1
        elif index != len(segments)-1:
            raise ValueError("A retained segment transition lacks its intervening edge gluing")
    if event_index != len(events) or time != elapsed:
        raise ValueError("Translation segments do not account for all events and elapsed time")
    final = data["final_state"]
    _keys(final, {"tile", "position", "pending_edges", "vertex_id"})
    _index(final["tile"], topology["tile_count"])
    _same(final["position"], list(map(str, position)), "Translation final position differs from the retained chain")
    if final["tile"] != tile:
        raise ValueError("Translation final tile differs from the retained chain")
    pending, vertex_id = _outgoing(position, direction), None
    if status == "completed":
        if remaining != 0 or pending:
            raise ValueError("Translation completion requires the full duration and all endpoint gluings")
    elif status == "stopped_at_vertex":
        if len(pending) != 2:
            raise ValueError("Translation vertex stop requires two simultaneous outgoing edges")
        corner = (2 if position[0] == 1 else 3) if position[1] == 1 else (1 if position[0] == 1 else 0)
        vertex_id = vertices[tile, corner]
    elif len(pending) != 1 or len(events) != request["max_events"]:
        raise ValueError("Translation budget stop requires an omitted single gluing after using the event budget")
    _same(final["pending_edges"], pending, "Translation pending-edge status differs")
    _same(final["vertex_id"], vertex_id, "Translation stopped vertex identity differs")
    expected_invariants = {name: status == "completed" if name == "requested_duration_completed" else True for name in INVARIANTS}
    _same(data["invariants"], expected_invariants, "Translation invariants or completion status differ")
