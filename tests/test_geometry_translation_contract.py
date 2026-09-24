"""Contract tests using retained records from the real TSDE run() implementation.

These frozen traces exercise CIW's offline content checks. They do not stand in
for the separate installed-provider/replay gate or authenticate fresh execution.
"""
from copy import deepcopy

import pytest

from ciw.geometry_translation_contract import validate_request, validate_result
from ciw.core.canonical import digest

# Captured from work/tsde-implementation/src/translation_surface_dynamics/flow.py.
# No provider import, subprocess, or numerical execution occurs in these tests.
RETAINED = {'complete': {'schema': 'tsde.square-tiled-flow-result.v1',
              'operation_id': 'tsde.square-tiled-flow.v1',
              'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                          'gluing': {'right': [0], 'up': [0]},
                          'start': {'tile': 0, 'position': ['1/4', '1/3']},
                          'direction': ['1', '1/2'],
                          'duration': '1',
                          'max_events': 4},
              'request_digest': 'sha256:4ccec63906800b25e92fa7ad81a4102209a61317ebeb26de242fcb9045a90635',
              'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
              'arithmetic': {'kind': 'exact_rational',
                             'coordinate_unit': 'unit_square_side',
                             'time_parameter': 'declared_flow_parameter',
                             'tolerance': '0',
                             'derivation': 'affine_position_and_rational_boundary_intersection',
                             'input_bits': 64,
                             'arithmetic_bits': 256,
                             'vertex_policy': 'stop_before_any_vertex_continuation',
                             'endpoint_policy': 'process_single_edge_crossing_at_duration',
                             'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
              'gluing_validation': {'permutations_bijective': True,
                                    'connected': True,
                                    'tile_count': 1,
                                    'edge_count': 2,
                                    'vertex_count': 1,
                                    'euler_characteristic': 0,
                                    'genus': 1,
                                    'right': [0],
                                    'left': [0],
                                    'up': [0],
                                    'down': [0],
                                    'vertices': [{'vertex_id': 0,
                                                  'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                                  'cone_angle_multiple_of_2pi': 1,
                                                  'singular': False}]},
              'status': 'completed',
              'elapsed': '1',
              'remaining': '0',
              'final_state': {'tile': 0, 'position': ['1/4', '5/6'], 'pending_edges': [], 'vertex_id': None},
              'segments': [{'tile': 0,
                            't_start': '0',
                            't_end': '3/4',
                            'start': ['1/4', '1/3'],
                            'end': ['1', '17/24']},
                           {'tile': 0,
                            't_start': '3/4',
                            't_end': '1',
                            'start': ['0', '17/24'],
                            'end': ['1/4', '5/6']}],
              'events': [{'index': 0,
                          'time': '3/4',
                          'edge': 'right',
                          'from_tile': 0,
                          'to_tile': 0,
                          'from_position': ['1', '17/24'],
                          'to_position': ['0', '17/24'],
                          'translation': ['-1', '0']}],
              'invariants': {'affine_segments_match_direction': True,
                             'segment_continuity_via_gluing': True,
                             'directed_edge_maps_match_permutations': True,
                             'positions_in_closed_unit_square': True,
                             'event_times_strictly_increasing': True,
                             'unfolded_displacement_matches': True,
                             'elapsed_within_requested_duration': True,
                             'requested_duration_completed': True},
              'artifact_digest': 'sha256:47c4801fa1a56d66a2b5fc64a83c14d2214a8116c4ae6a88b2b8ff0f5fddfc13'},
 'corner': {'schema': 'tsde.square-tiled-flow-result.v1',
            'operation_id': 'tsde.square-tiled-flow.v1',
            'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                        'gluing': {'right': [0], 'up': [0]},
                        'start': {'tile': 0, 'position': ['1/4', '1/4']},
                        'direction': ['1', '1'],
                        'duration': '3/4',
                        'max_events': 4},
            'request_digest': 'sha256:3354fd370f45c5dae6a542eb3dc9476cdf64056baad468c2dd0c7f215193e453',
            'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
            'arithmetic': {'kind': 'exact_rational',
                           'coordinate_unit': 'unit_square_side',
                           'time_parameter': 'declared_flow_parameter',
                           'tolerance': '0',
                           'derivation': 'affine_position_and_rational_boundary_intersection',
                           'input_bits': 64,
                           'arithmetic_bits': 256,
                           'vertex_policy': 'stop_before_any_vertex_continuation',
                           'endpoint_policy': 'process_single_edge_crossing_at_duration',
                           'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
            'gluing_validation': {'permutations_bijective': True,
                                  'connected': True,
                                  'tile_count': 1,
                                  'edge_count': 2,
                                  'vertex_count': 1,
                                  'euler_characteristic': 0,
                                  'genus': 1,
                                  'right': [0],
                                  'left': [0],
                                  'up': [0],
                                  'down': [0],
                                  'vertices': [{'vertex_id': 0,
                                                'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                                'cone_angle_multiple_of_2pi': 1,
                                                'singular': False}]},
            'status': 'stopped_at_vertex',
            'elapsed': '3/4',
            'remaining': '0',
            'final_state': {'tile': 0, 'position': ['1', '1'], 'pending_edges': ['right', 'up'], 'vertex_id': 0},
            'segments': [{'tile': 0, 't_start': '0', 't_end': '3/4', 'start': ['1/4', '1/4'], 'end': ['1', '1']}],
            'events': [],
            'invariants': {'affine_segments_match_direction': True,
                           'segment_continuity_via_gluing': True,
                           'directed_edge_maps_match_permutations': True,
                           'positions_in_closed_unit_square': True,
                           'event_times_strictly_increasing': True,
                           'unfolded_displacement_matches': True,
                           'elapsed_within_requested_duration': True,
                           'requested_duration_completed': False},
            'artifact_digest': 'sha256:fd1c4cb5c5b8b39d087a4ad3dc710455fb000e7e1adb5e1b7f5719663844d3db'},
 'budget_endpoint': {'schema': 'tsde.square-tiled-flow-result.v1',
                     'operation_id': 'tsde.square-tiled-flow.v1',
                     'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                                 'gluing': {'right': [0], 'up': [0]},
                                 'start': {'tile': 0, 'position': ['1/4', '1/3']},
                                 'direction': ['1', '0'],
                                 'duration': '3/4',
                                 'max_events': 0},
                     'request_digest': 'sha256:a6dd60109a8e83bb5a4f779654c6ccd1387d7f3ecf94a17855a1fe383e1fdc68',
                     'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
                     'arithmetic': {'kind': 'exact_rational',
                                    'coordinate_unit': 'unit_square_side',
                                    'time_parameter': 'declared_flow_parameter',
                                    'tolerance': '0',
                                    'derivation': 'affine_position_and_rational_boundary_intersection',
                                    'input_bits': 64,
                                    'arithmetic_bits': 256,
                                    'vertex_policy': 'stop_before_any_vertex_continuation',
                                    'endpoint_policy': 'process_single_edge_crossing_at_duration',
                                    'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
                     'gluing_validation': {'permutations_bijective': True,
                                           'connected': True,
                                           'tile_count': 1,
                                           'edge_count': 2,
                                           'vertex_count': 1,
                                           'euler_characteristic': 0,
                                           'genus': 1,
                                           'right': [0],
                                           'left': [0],
                                           'up': [0],
                                           'down': [0],
                                           'vertices': [{'vertex_id': 0,
                                                         'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                                         'cone_angle_multiple_of_2pi': 1,
                                                         'singular': False}]},
                     'status': 'event_budget_exhausted',
                     'elapsed': '3/4',
                     'remaining': '0',
                     'final_state': {'tile': 0,
                                     'position': ['1', '1/3'],
                                     'pending_edges': ['right'],
                                     'vertex_id': None},
                     'segments': [{'tile': 0,
                                   't_start': '0',
                                   't_end': '3/4',
                                   'start': ['1/4', '1/3'],
                                   'end': ['1', '1/3']}],
                     'events': [],
                     'invariants': {'affine_segments_match_direction': True,
                                    'segment_continuity_via_gluing': True,
                                    'directed_edge_maps_match_permutations': True,
                                    'positions_in_closed_unit_square': True,
                                    'event_times_strictly_increasing': True,
                                    'unfolded_displacement_matches': True,
                                    'elapsed_within_requested_duration': True,
                                    'requested_duration_completed': False},
                     'artifact_digest': 'sha256:e56579438ad03da55afee0d4b953c06da5a72985a1624cda3c12bc49ffaeae9a'},
 'budget_prefix': {'schema': 'tsde.square-tiled-flow-result.v1',
                   'operation_id': 'tsde.square-tiled-flow.v1',
                   'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                               'gluing': {'right': [0], 'up': [0]},
                               'start': {'tile': 0, 'position': ['1/4', '1/3']},
                               'direction': ['1', '0'],
                               'duration': '2',
                               'max_events': 1},
                   'request_digest': 'sha256:846d351e504e28fdd66a8f6b77782e758a8daca26e819943673f4babc31d127f',
                   'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
                   'arithmetic': {'kind': 'exact_rational',
                                  'coordinate_unit': 'unit_square_side',
                                  'time_parameter': 'declared_flow_parameter',
                                  'tolerance': '0',
                                  'derivation': 'affine_position_and_rational_boundary_intersection',
                                  'input_bits': 64,
                                  'arithmetic_bits': 256,
                                  'vertex_policy': 'stop_before_any_vertex_continuation',
                                  'endpoint_policy': 'process_single_edge_crossing_at_duration',
                                  'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
                   'gluing_validation': {'permutations_bijective': True,
                                         'connected': True,
                                         'tile_count': 1,
                                         'edge_count': 2,
                                         'vertex_count': 1,
                                         'euler_characteristic': 0,
                                         'genus': 1,
                                         'right': [0],
                                         'left': [0],
                                         'up': [0],
                                         'down': [0],
                                         'vertices': [{'vertex_id': 0,
                                                       'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                                       'cone_angle_multiple_of_2pi': 1,
                                                       'singular': False}]},
                   'status': 'event_budget_exhausted',
                   'elapsed': '7/4',
                   'remaining': '1/4',
                   'final_state': {'tile': 0,
                                   'position': ['1', '1/3'],
                                   'pending_edges': ['right'],
                                   'vertex_id': None},
                   'segments': [{'tile': 0,
                                 't_start': '0',
                                 't_end': '3/4',
                                 'start': ['1/4', '1/3'],
                                 'end': ['1', '1/3']},
                                {'tile': 0,
                                 't_start': '3/4',
                                 't_end': '7/4',
                                 'start': ['0', '1/3'],
                                 'end': ['1', '1/3']}],
                   'events': [{'index': 0,
                               'time': '3/4',
                               'edge': 'right',
                               'from_tile': 0,
                               'to_tile': 0,
                               'from_position': ['1', '1/3'],
                               'to_position': ['0', '1/3'],
                               'translation': ['-1', '0']}],
                   'invariants': {'affine_segments_match_direction': True,
                                  'segment_continuity_via_gluing': True,
                                  'directed_edge_maps_match_permutations': True,
                                  'positions_in_closed_unit_square': True,
                                  'event_times_strictly_increasing': True,
                                  'unfolded_displacement_matches': True,
                                  'elapsed_within_requested_duration': True,
                                  'requested_duration_completed': False},
                   'artifact_digest': 'sha256:b3e191a37c118e1e5316649d0b6ff91af3bfdd4fcdd6422b2a2ff8002dbcd8e4'},
 'zero': {'schema': 'tsde.square-tiled-flow-result.v1',
          'operation_id': 'tsde.square-tiled-flow.v1',
          'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                      'gluing': {'right': [0], 'up': [0]},
                      'start': {'tile': 0, 'position': ['1/4', '1/3']},
                      'direction': ['1', '1/2'],
                      'duration': '0',
                      'max_events': 0},
          'request_digest': 'sha256:c671c89b2a661c166a334df29a261b669a1b8ad69fa1c5ce175493c7d1ab59d1',
          'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
          'arithmetic': {'kind': 'exact_rational',
                         'coordinate_unit': 'unit_square_side',
                         'time_parameter': 'declared_flow_parameter',
                         'tolerance': '0',
                         'derivation': 'affine_position_and_rational_boundary_intersection',
                         'input_bits': 64,
                         'arithmetic_bits': 256,
                         'vertex_policy': 'stop_before_any_vertex_continuation',
                         'endpoint_policy': 'process_single_edge_crossing_at_duration',
                         'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
          'gluing_validation': {'permutations_bijective': True,
                                'connected': True,
                                'tile_count': 1,
                                'edge_count': 2,
                                'vertex_count': 1,
                                'euler_characteristic': 0,
                                'genus': 1,
                                'right': [0],
                                'left': [0],
                                'up': [0],
                                'down': [0],
                                'vertices': [{'vertex_id': 0,
                                              'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                              'cone_angle_multiple_of_2pi': 1,
                                              'singular': False}]},
          'status': 'completed',
          'elapsed': '0',
          'remaining': '0',
          'final_state': {'tile': 0, 'position': ['1/4', '1/3'], 'pending_edges': [], 'vertex_id': None},
          'segments': [],
          'events': [],
          'invariants': {'affine_segments_match_direction': True,
                         'segment_continuity_via_gluing': True,
                         'directed_edge_maps_match_permutations': True,
                         'positions_in_closed_unit_square': True,
                         'event_times_strictly_increasing': True,
                         'unfolded_displacement_matches': True,
                         'elapsed_within_requested_duration': True,
                         'requested_duration_completed': True},
          'artifact_digest': 'sha256:b1b46890278a3a31135d3a944b2da3f891a5ea5c799fee082e50a0f88a2b33c1'},
 'negative': {'schema': 'tsde.square-tiled-flow-result.v1',
              'operation_id': 'tsde.square-tiled-flow.v1',
              'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                          'gluing': {'right': [1, 2, 0], 'up': [0, 1, 2]},
                          'start': {'tile': 0, 'position': ['1/4', '1/3']},
                          'direction': ['-1', '0'],
                          'duration': '1/2',
                          'max_events': 4},
              'request_digest': 'sha256:c9c7df8808cf0eabf97dd56daeea77ace59011046df65bf7b0b190824e1a9d57',
              'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
              'arithmetic': {'kind': 'exact_rational',
                             'coordinate_unit': 'unit_square_side',
                             'time_parameter': 'declared_flow_parameter',
                             'tolerance': '0',
                             'derivation': 'affine_position_and_rational_boundary_intersection',
                             'input_bits': 64,
                             'arithmetic_bits': 256,
                             'vertex_policy': 'stop_before_any_vertex_continuation',
                             'endpoint_policy': 'process_single_edge_crossing_at_duration',
                             'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
              'gluing_validation': {'permutations_bijective': True,
                                    'connected': True,
                                    'tile_count': 3,
                                    'edge_count': 6,
                                    'vertex_count': 3,
                                    'euler_characteristic': 0,
                                    'genus': 1,
                                    'right': [1, 2, 0],
                                    'left': [2, 0, 1],
                                    'up': [0, 1, 2],
                                    'down': [0, 1, 2],
                                    'vertices': [{'vertex_id': 0,
                                                  'corners': [[0, 0], [0, 3], [2, 1], [2, 2]],
                                                  'cone_angle_multiple_of_2pi': 1,
                                                  'singular': False},
                                                 {'vertex_id': 1,
                                                  'corners': [[0, 1], [0, 2], [1, 0], [1, 3]],
                                                  'cone_angle_multiple_of_2pi': 1,
                                                  'singular': False},
                                                 {'vertex_id': 2,
                                                  'corners': [[1, 1], [1, 2], [2, 0], [2, 3]],
                                                  'cone_angle_multiple_of_2pi': 1,
                                                  'singular': False}]},
              'status': 'completed',
              'elapsed': '1/2',
              'remaining': '0',
              'final_state': {'tile': 2, 'position': ['3/4', '1/3'], 'pending_edges': [], 'vertex_id': None},
              'segments': [{'tile': 0,
                            't_start': '0',
                            't_end': '1/4',
                            'start': ['1/4', '1/3'],
                            'end': ['0', '1/3']},
                           {'tile': 2,
                            't_start': '1/4',
                            't_end': '1/2',
                            'start': ['1', '1/3'],
                            'end': ['3/4', '1/3']}],
              'events': [{'index': 0,
                          'time': '1/4',
                          'edge': 'left',
                          'from_tile': 0,
                          'to_tile': 2,
                          'from_position': ['0', '1/3'],
                          'to_position': ['1', '1/3'],
                          'translation': ['1', '0']}],
              'invariants': {'affine_segments_match_direction': True,
                             'segment_continuity_via_gluing': True,
                             'directed_edge_maps_match_permutations': True,
                             'positions_in_closed_unit_square': True,
                             'event_times_strictly_increasing': True,
                             'unfolded_displacement_matches': True,
                             'elapsed_within_requested_duration': True,
                             'requested_duration_completed': True},
              'artifact_digest': 'sha256:37f2f21c183dc91f9d1164dd1b34ad5a66a4356b6dfbc0973b615ceb95244ec0'},
 'genus_two': {'schema': 'tsde.square-tiled-flow-result.v1',
               'operation_id': 'tsde.square-tiled-flow.v1',
               'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                           'gluing': {'right': [1, 0, 2], 'up': [2, 1, 0]},
                           'start': {'tile': 0, 'position': ['1/4', '1/3']},
                           'direction': ['1', '1/2'],
                           'duration': '3',
                           'max_events': 128},
               'request_digest': 'sha256:f3af77d91bfffc2fd27483ab49edbb07870898de1277a83ca5d188cd292e61be',
               'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
               'arithmetic': {'kind': 'exact_rational',
                              'coordinate_unit': 'unit_square_side',
                              'time_parameter': 'declared_flow_parameter',
                              'tolerance': '0',
                              'derivation': 'affine_position_and_rational_boundary_intersection',
                              'input_bits': 64,
                              'arithmetic_bits': 256,
                              'vertex_policy': 'stop_before_any_vertex_continuation',
                              'endpoint_policy': 'process_single_edge_crossing_at_duration',
                              'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
               'gluing_validation': {'permutations_bijective': True,
                                     'connected': True,
                                     'tile_count': 3,
                                     'edge_count': 6,
                                     'vertex_count': 1,
                                     'euler_characteristic': -2,
                                     'genus': 2,
                                     'right': [1, 0, 2],
                                     'left': [1, 0, 2],
                                     'up': [2, 1, 0],
                                     'down': [2, 1, 0],
                                     'vertices': [{'vertex_id': 0,
                                                   'corners': [[0, 0],
                                                               [0, 1],
                                                               [0, 2],
                                                               [0, 3],
                                                               [1, 0],
                                                               [1, 1],
                                                               [1, 2],
                                                               [1, 3],
                                                               [2, 0],
                                                               [2, 1],
                                                               [2, 2],
                                                               [2, 3]],
                                                   'cone_angle_multiple_of_2pi': 3,
                                                   'singular': True}]},
               'status': 'completed',
               'elapsed': '3',
               'remaining': '0',
               'final_state': {'tile': 1, 'position': ['1/4', '5/6'], 'pending_edges': [], 'vertex_id': None},
               'segments': [{'tile': 0,
                             't_start': '0',
                             't_end': '3/4',
                             'start': ['1/4', '1/3'],
                             'end': ['1', '17/24']},
                            {'tile': 1,
                             't_start': '3/4',
                             't_end': '4/3',
                             'start': ['0', '17/24'],
                             'end': ['7/12', '1']},
                            {'tile': 1,
                             't_start': '4/3',
                             't_end': '7/4',
                             'start': ['7/12', '0'],
                             'end': ['1', '5/24']},
                            {'tile': 0,
                             't_start': '7/4',
                             't_end': '11/4',
                             'start': ['0', '5/24'],
                             'end': ['1', '17/24']},
                            {'tile': 1,
                             't_start': '11/4',
                             't_end': '3',
                             'start': ['0', '17/24'],
                             'end': ['1/4', '5/6']}],
               'events': [{'index': 0,
                           'time': '3/4',
                           'edge': 'right',
                           'from_tile': 0,
                           'to_tile': 1,
                           'from_position': ['1', '17/24'],
                           'to_position': ['0', '17/24'],
                           'translation': ['-1', '0']},
                          {'index': 1,
                           'time': '4/3',
                           'edge': 'up',
                           'from_tile': 1,
                           'to_tile': 1,
                           'from_position': ['7/12', '1'],
                           'to_position': ['7/12', '0'],
                           'translation': ['0', '-1']},
                          {'index': 2,
                           'time': '7/4',
                           'edge': 'right',
                           'from_tile': 1,
                           'to_tile': 0,
                           'from_position': ['1', '5/24'],
                           'to_position': ['0', '5/24'],
                           'translation': ['-1', '0']},
                          {'index': 3,
                           'time': '11/4',
                           'edge': 'right',
                           'from_tile': 0,
                           'to_tile': 1,
                           'from_position': ['1', '17/24'],
                           'to_position': ['0', '17/24'],
                           'translation': ['-1', '0']}],
               'invariants': {'affine_segments_match_direction': True,
                              'segment_continuity_via_gluing': True,
                              'directed_edge_maps_match_permutations': True,
                              'positions_in_closed_unit_square': True,
                              'event_times_strictly_increasing': True,
                              'unfolded_displacement_matches': True,
                              'elapsed_within_requested_duration': True,
                              'requested_duration_completed': True},
               'artifact_digest': 'sha256:dd0db9257e259428ea1fc7c76c3f6e0719215f61f7d08714ad28b312d5e9e4c2'},
 'single_edge_endpoint': {'schema': 'tsde.square-tiled-flow-result.v1',
                          'operation_id': 'tsde.square-tiled-flow.v1',
                          'request': {'schema': 'tsde.square-tiled-flow-request.v1',
                                      'gluing': {'right': [0], 'up': [0]},
                                      'start': {'tile': 0, 'position': ['1/4', '1/3']},
                                      'direction': ['0', '-1'],
                                      'duration': '1/3',
                                      'max_events': 4},
                          'request_digest': 'sha256:51c7dbac7d95b9d6d568813c714e40f83247458c3eea0b1e6dea949779ea9c80',
                          'claim_scope': 'exact_rational_translation_flow_prefix_on_declared_square_tiled_surface',
                          'arithmetic': {'kind': 'exact_rational',
                                         'coordinate_unit': 'unit_square_side',
                                         'time_parameter': 'declared_flow_parameter',
                                         'tolerance': '0',
                                         'derivation': 'affine_position_and_rational_boundary_intersection',
                                         'input_bits': 64,
                                         'arithmetic_bits': 256,
                                         'vertex_policy': 'stop_before_any_vertex_continuation',
                                         'endpoint_policy': 'process_single_edge_crossing_at_duration',
                                         'event_budget_policy': 'stop_at_next_boundary_before_omitted_gluing'},
                          'gluing_validation': {'permutations_bijective': True,
                                                'connected': True,
                                                'tile_count': 1,
                                                'edge_count': 2,
                                                'vertex_count': 1,
                                                'euler_characteristic': 0,
                                                'genus': 1,
                                                'right': [0],
                                                'left': [0],
                                                'up': [0],
                                                'down': [0],
                                                'vertices': [{'vertex_id': 0,
                                                              'corners': [[0, 0], [0, 1], [0, 2], [0, 3]],
                                                              'cone_angle_multiple_of_2pi': 1,
                                                              'singular': False}]},
                          'status': 'completed',
                          'elapsed': '1/3',
                          'remaining': '0',
                          'final_state': {'tile': 0,
                                          'position': ['1/4', '1'],
                                          'pending_edges': [],
                                          'vertex_id': None},
                          'segments': [{'tile': 0,
                                        't_start': '0',
                                        't_end': '1/3',
                                        'start': ['1/4', '1/3'],
                                        'end': ['1/4', '0']}],
                          'events': [{'index': 0,
                                      'time': '1/3',
                                      'edge': 'down',
                                      'from_tile': 0,
                                      'to_tile': 0,
                                      'from_position': ['1/4', '0'],
                                      'to_position': ['1/4', '1'],
                                      'translation': ['0', '1']}],
                          'invariants': {'affine_segments_match_direction': True,
                                         'segment_continuity_via_gluing': True,
                                         'directed_edge_maps_match_permutations': True,
                                         'positions_in_closed_unit_square': True,
                                         'event_times_strictly_increasing': True,
                                         'unfolded_displacement_matches': True,
                                         'elapsed_within_requested_duration': True,
                                         'requested_duration_completed': True},
                          'artifact_digest': 'sha256:50968d0102182463885ce66a1246c7288a48764f940b9854214df7e6d84a646e'}}


def trace(name="complete"):
    return deepcopy(RETAINED[name])


def reseal(data):
    data["artifact_digest"] = digest({key: value for key, value in data.items() if key != "artifact_digest"})
    return data


def set_path(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize("name", sorted(RETAINED))
def test_accepts_retained_native_record(name):
    data = trace(name)
    request = deepcopy(data["request"])
    before = deepcopy(data)
    assert validate_request(request) == request
    validate_result(request, data)
    assert data == before
    assert request == before["request"]


def test_request_copy_has_no_caller_owned_aliases():
    request = trace()["request"]
    validated = validate_request(request)
    validated["gluing"]["right"][0] = 99
    assert request["gluing"]["right"] == [0]


@pytest.mark.parametrize("path,value", [
    (("schema",), "other"),
    (("gluing", "right"), []),
    (("gluing", "up"), [0, 1]),
    (("gluing", "right"), [True]),
    (("gluing", "right"), [1]),
    (("gluing",), {"right": [0, 1], "up": [0, 1]}),
    (("gluing",), {"right": list(range(1, 33)) + [0], "up": list(range(33))}),
    (("start", "tile"), True),
    (("start", "tile"), -1),
    (("start", "position"), ["0", "1/3"]),
    (("start", "position"), ["1", "1/3"]),
    (("start", "position"), ["2/4", "1/3"]),
    (("start", "position"), ["0.25", "1/3"]),
    (("start", "position"), ["1/18446744073709551616", "1/3"]),
    (("start", "position"), ["1/0", "1/3"]),
    (("direction",), ["0", "0"]),
    (("direction",), ["1025", "0"]),
    (("direction",), ["-1025", "0"]),
    (("direction",), ["1", "1", "1"]),
    (("direction",), [True, "1"]),
    (("direction",), ["1" * 44, "1"]),
    (("duration",), "1025"),
    (("duration",), "-1"),
    (("duration",), "+1"),
    (("duration",), "1/1"),
    (("duration",), "NaN"),
    (("max_events",), True),
    (("max_events",), 1.0),
    (("max_events",), -1),
    (("max_events",), 257),
])
def test_request_profile_rejects_unsupported_values(path, value):
    request = trace()["request"]
    set_path(request, path, value)
    with pytest.raises(ValueError):
        validate_request(request)


@pytest.mark.parametrize("value", [object(), float("inf"), b"1"])
def test_non_json_request_is_value_error(value):
    request = trace()["request"]
    request["duration"] = value
    with pytest.raises(ValueError):
        validate_request(request)


@pytest.mark.parametrize("path,value", [
    (("schema",), "other"),
    (("operation_id",), "other"),
    (("claim_scope",), "unbounded_continuous_flow"),
    (("request", "duration"), "2"),
    (("request_digest",), "sha256:" + "0" * 64),
    (("arithmetic", "vertex_policy"), "continue_through_vertices"),
    (("arithmetic", "endpoint_policy"), "omit_endpoint_gluing"),
    (("arithmetic", "event_budget_policy"), "complete_after_budget"),
    (("arithmetic", "arithmetic_bits"), 257),
    (("arithmetic", "input_bits"), 64.0),
    (("arithmetic", "tolerance"), "1/100"),
    (("gluing_validation", "genus"), 999),
    (("gluing_validation", "connected"), 1),
    (("gluing_validation", "right"), [1]),
    (("gluing_validation", "left"), [1]),
    (("gluing_validation", "vertices", 0, "cone_angle_multiple_of_2pi"), 3),
    (("gluing_validation", "vertices", 0, "corners"), [[0, 0]]),
    (("gluing_validation", "vertices", 0, "singular"), True),
    (("elapsed",), "2"),
    (("remaining",), "1"),
    (("elapsed",), "1/1"),
    (("status",), "unknown"),
    (("segments",), [{"invented": True}]),
    (("segments",), []),
    (("segments", 0, "tile"), True),
    (("segments", 0, "t_start"), "1/100"),
    (("segments", 0, "t_end"), "0"),
    (("segments", 0, "start"), ["1/3", "1/3"]),
    (("segments", 0, "end"), ["1", "2/3"]),
    (("segments", 0, "end"), ["2", "17/24"]),
    (("segments", 1, "t_start"), "1/2"),
    (("events",), []),
    (("events", 0, "index"), True),
    (("events", 0, "time"), "1/2"),
    (("events", 0, "edge"), "up"),
    (("events", 0, "from_tile"), 1),
    (("events", 0, "to_tile"), 1),
    (("events", 0, "from_position"), ["1", "1/3"]),
    (("events", 0, "to_position"), ["0", "1/3"]),
    (("events", 0, "translation"), ["0", "0"]),
    (("final_state", "tile"), True),
    (("final_state", "position"), ["0", "0"]),
    (("final_state", "pending_edges"), ["right"]),
    (("final_state", "vertex_id"), 0),
    (("invariants", "unfolded_displacement_matches"), False),
    (("invariants", "requested_duration_completed"), 1),
])
def test_resealed_content_cannot_bypass_record_bindings(path, value):
    data = trace()
    set_path(data, path, value)
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(RETAINED["complete"]["request"], data)


@pytest.mark.parametrize("name", ["corner", "budget_endpoint", "budget_prefix"])
def test_stopped_prefix_cannot_be_relabeled_complete_even_at_requested_end(name):
    data = trace(name)
    data["status"] = "completed"
    data["invariants"]["requested_duration_completed"] = True
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_corner_cannot_be_labeled_budget_stop():
    data = trace("corner")
    data["request"]["max_events"] = 0
    data["request_digest"] = digest(data["request"])
    data["status"] = "event_budget_exhausted"
    data["final_state"]["vertex_id"] = None
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_budget_stop_requires_exhausted_budget():
    data = trace("budget_endpoint")
    data["request"]["max_events"] = 1
    data["request_digest"] = digest(data["request"])
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_inverse_gluing_is_not_forward_permutation():
    data = trace("negative")
    assert data["events"][0]["to_tile"] == 2
    data["gluing_validation"]["left"] = data["gluing_validation"]["right"][:]
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_genus_two_corner_class_cannot_be_replaced_with_regular_vertices():
    data = trace("genus_two")
    assert data["gluing_validation"]["genus"] == 2
    data["gluing_validation"]["vertices"] = trace()["gluing_validation"]["vertices"]
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_event_order_and_count_are_checked():
    data = trace("genus_two")
    assert len(data["events"]) > 1
    data["events"] = list(reversed(data["events"]))
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


@pytest.mark.parametrize("field", ["segments", "events"])
def test_unaccounted_extra_records_are_rejected(field):
    data = trace()
    data[field].append(deepcopy(data[field][0]))
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


def test_rational_output_budget_is_checked():
    data = trace()
    data["segments"][0]["t_end"] = "1/" + str(1 << 256)
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


@pytest.mark.parametrize("path", [(), ("request",), ("arithmetic",), ("final_state",), ("segments", 0), ("events", 0)])
def test_unexpected_fields_are_rejected(path):
    data = trace()
    target = data
    for key in path:
        target = target[key]
    target["unexpected"] = True
    reseal(data)
    with pytest.raises(ValueError):
        validate_result(RETAINED["complete"]["request"], data)


def test_stale_identity_is_rejected_without_resealing():
    data = trace()
    data["artifact_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        validate_result(data["request"], data)


@pytest.mark.parametrize("value", [object(), float("inf"), b"1"])
def test_non_json_result_is_value_error(value):
    data = trace()
    data["elapsed"] = value
    with pytest.raises(ValueError):
        validate_result(data["request"], data)
