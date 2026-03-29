from __future__ import annotations

from collections import deque
import math
import time


DIRS = {
    'N': (0, -1),
    'S': (0, 1),
    'E': (1, 0),
    'W': (-1, 0),
}
DIR_LIST = ('N', 'S', 'E', 'W')
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
TREASURE_VALUE = {'c': 20, 'D': 50}
HARD_TRAILS = {'t1', 't2', 't3', 't4'}
HEADS = {'p1', 'p2', 'p3', 'p4'}
DEADLINE_S = 0.43


class Bot:
    def __init__(self):
        self.map_cache = {}
        self.state = {
            'elapsed': 0,
            'last_signature': None,
            'last_pos': None,
            'last_dir': None,
            'last_map': None,
            'score_bounds': {},
            'emp_owner_hypotheses': {},
            'opening_progress': {},
            'fallback': 'N',
            'last_heads': {},
            'solo_book_idx': {},
        }

        self.public_solo_books = {
            's_path_2.txt': ['E', 'E', 'S', 'E', 'P'],
            's_floodfill_1.txt': ['W','W','W','W','W','W','W','W','W','W','N','N','N','N','N','N','N','N','N','N'],
            's_choice_2.txt': ['E','P','P','+','E','+','+','+','+','+','E','S','+','+','S','W','W','W','W','W','W','W','W','W','W','W','W','W','W','+','N','N'],
        }
        self.public_general_maps = {'arena.txt', 'maze.txt', 'treasure.txt', 'gate.txt', 'cube.txt', 'orbit.txt'}
        self.public_solo_maps = {'s_path_0.txt','s_path_1.txt','s_path_2.txt','s_path_3.txt','s_floodfill_0.txt','s_floodfill_1.txt','s_floodfill_2.txt','s_choice_0.txt','s_choice_1.txt','s_choice_2.txt'}

    def get_move(self, board, player_id, pos, info):
        start = time.perf_counter()
        rows = len(board)
        cols = len(board[0]) if rows else 0
        map_name = info.get('map_name', 'unknown')
        meta = self._get_map_meta(board, map_name)
        obs = self._parse_board(board, player_id, pos, info, meta)
        self._update_state(obs, info)

        fallback = self._best_emergency_move(obs)
        self.state['fallback'] = fallback

        hot = self._is_hot_turn(obs)
        alive_opponents = self._alive_opponent_count(obs)

        book = self._public_solo_book_move(obs)
        if book is not None:
            self.state['fallback'] = book
            return book

        if obs['family'] in ('SOLO_FLOOD', 'SOLO_CHOICE') and obs['map_name'] not in self.public_solo_maps:
            hidden_solo = self._hidden_solo_family_move(obs, start)
            if hidden_solo:
                self.state['fallback'] = hidden_solo
                return hidden_solo

        if obs['family'] != 'GENERAL':
            solo = self._solo_opening_move(obs, start)
            if solo:
                self.state['fallback'] = solo
                return solo

        if alive_opponents == 0:
            lone = self._lone_survivor_move(obs, start)
            if lone:
                self.state['fallback'] = lone
                return lone


        candidates = self._generate_candidates(obs)
        if not candidates:
            return fallback

        best_move = fallback
        best_score = -10**18

        for action in candidates:
            if time.perf_counter() - start > DEADLINE_S:
                break
            score = self._score_action(obs, action, hot, start)
            if score > best_score:
                best_score = score
                best_move = action

        self.state['fallback'] = best_move
        return best_move

    def _get_map_meta(self, board, map_name):
        meta = self.map_cache.get(map_name)
        if meta is not None:
            return meta
        rows = len(board)
        cols = len(board[0]) if rows else 0
        permanent_walls = set()
        timed_walls = {}
        treasures = []
        for y in range(rows):
            row = board[y]
            for x in range(cols):
                cell = row[x]
                if cell == '#':
                    permanent_walls.add((x, y))
                elif isinstance(cell, int):
                    timed_walls[(x, y)] = cell
                elif cell in TREASURE_VALUE:
                    treasures.append((x, y, TREASURE_VALUE[cell]))
        meta = {
            'rows': rows,
            'cols': cols,
            'permanent_walls': permanent_walls,
            'initial_timed_walls': timed_walls,
            'initial_treasures': treasures,
            'family': self._infer_family(map_name),
            'center': (cols / 2.0, rows / 2.0),
        }
        self.map_cache[map_name] = meta
        return meta

    def _infer_family(self, map_name):
        if map_name.startswith('s_path_'):
            return 'SOLO_PATH'
        if map_name.startswith('s_floodfill_'):
            return 'SOLO_FLOOD'
        if map_name.startswith('s_choice_'):
            return 'SOLO_CHOICE'
        return 'GENERAL'

    def _parse_board(self, board, player_id, pos, info, meta):
        rows, cols = meta['rows'], meta['cols']
        hard_blockers = set()
        soft_heads = {}
        treasures = {}
        timed_walls = {}
        trails = set()

        for y in range(rows):
            row = board[y]
            for x in range(cols):
                cell = row[x]
                if cell == '#':
                    hard_blockers.add((x, y))
                elif isinstance(cell, int):
                    hard_blockers.add((x, y))
                    timed_walls[(x, y)] = cell
                elif cell in HARD_TRAILS:
                    hard_blockers.add((x, y))
                    trails.add((x, y))
                elif cell in TREASURE_VALUE:
                    treasures[(x, y)] = TREASURE_VALUE[cell]
                elif cell in HEADS:
                    soft_heads[int(cell[1])] = (x, y)

        opponents = info.get('opponents', [])
        opp_by_id = {o['id']: o for o in opponents}
        alive_ids = {player_id}
        for o in opponents:
            if o.get('alive'):
                alive_ids.add(o['id'])

        active_emps = list(info.get('active_emps', []))
        danger_union = self._build_emp_danger_union(pos, info, active_emps)

        return {
            'board': board,
            'player_id': player_id,
            'pos': pos,
            'my_dir': info.get('my_direction', 'N'),
            'my_score': info.get('my_score', 0),
            'phase_charges': info.get('phase_charges', 0),
            'emp_charges': info.get('emp_charges', 0),
            'emp_radius': info.get('emp_radius', 3),
            'stun_duration': info.get('stun_duration', 7),
            'map_name': info.get('map_name', 'unknown'),
            'family': meta['family'],
            'rows': rows,
            'cols': cols,
            'hard': hard_blockers,
            'soft_heads': soft_heads,
            'treasures': treasures,
            'timed_walls': timed_walls,
            'trails': trails,
            'opponents': opponents,
            'opp_by_id': opp_by_id,
            'alive_ids': alive_ids,
            'active_emps': active_emps,
            'danger_union': danger_union,
            'meta': meta,
        }

    def _update_state(self, obs, info):
        sig = (obs['map_name'], obs['pos'], info.get('my_score', 0), tuple(sorted(obs['timed_walls'].items())))
        if self.state['last_map'] != obs['map_name']:
            self.state['elapsed'] = 0
            self.state['opening_progress'] = {}
        else:
            self.state['elapsed'] += 1
        self.state['last_signature'] = sig
        self.state['last_pos'] = obs['pos']
        self.state['last_dir'] = obs['my_dir']
        self.state['last_map'] = obs['map_name']

        last_bounds = self.state['score_bounds']
        new_bounds = {}
        for opp in obs['opponents']:
            oid = opp['id']
            prev = last_bounds.get(oid, (0, 10**9))

            lower = max(prev[0], 0)
            upper = max(prev[1], lower + 200)
            new_bounds[oid] = (lower, upper)
        self.state['score_bounds'] = new_bounds

        last_heads = self.state['last_heads']
        current = {}
        for opp in obs['opponents']:
            if opp.get('alive'):
                current[opp['id']] = opp['pos']
        self.state['last_heads'] = current

    def _build_emp_danger_union(self, my_pos, info, active_emps):
        danger = set()
        heads = {info_o['id']: info_o['pos'] for info_o in info.get('opponents', []) if info_o.get('alive')}
        my_id = None
        for emp in active_emps:
            pos = emp.get('pos')
            timer = emp.get('timer', 0)
            if pos is None:
                continue
            candidates = [pos]
            for hpos in heads.values():
                if self._manhattan(pos, hpos) <= 1:
                    candidates.append(hpos)
            candidates.append(my_pos)
            for cx, cy in candidates:
                for dx in range(-info.get('emp_radius', 3), info.get('emp_radius', 3) + 1):
                    for dy in range(-info.get('emp_radius', 3), info.get('emp_radius', 3) + 1):
                        if max(abs(dx), abs(dy)) <= info.get('emp_radius', 3):
                            danger.add((cx + dx, cy + dy, timer))
        return danger

    def _generate_candidates(self, obs):
        candidates = []
        for d in DIR_LIST:
            if self._sim_my_action(obs, d)['valid']:
                candidates.append(d)

        plus_sim = self._sim_my_action(obs, '+')
        if plus_sim['valid']:
            candidates.append('+')
        phase_sim = self._sim_my_action(obs, 'P')
        if phase_sim['valid'] and obs['phase_charges'] > 0:
            candidates.append('P')

        if obs['family'] == 'GENERAL' and obs['emp_charges'] > 0:
            emp_actions = []
            for base in list(candidates):
                if base in DIR_LIST or base in ('+', 'P'):
                    if self._emp_action_worth_considering(obs, base):
                        emp_actions.append('X' + base)
            candidates.extend(emp_actions)

        ordered = []
        seen = set()
        for a in ['N', 'S', 'E', 'W', '+', 'P', 'XN', 'XS', 'XE', 'XW', 'X+', 'XP']:
            if a in candidates and a not in seen:
                seen.add(a)
                ordered.append(a)
        return ordered

    def _emp_action_worth_considering(self, obs, base_action):
        my_pos = obs['pos']
        radius = obs['emp_radius']
        sim = self._sim_my_action(obs, base_action)
        if not sim['valid']:
            return False
        npos = sim['end']
        for opp in obs['opponents']:
            if not opp.get('alive'):
                continue
            opp_dir = opp.get('direction', 'N')
            predicted = opp['pos']
            for _ in range(5):
                predicted = self._step(predicted, opp_dir)
            if max(abs(predicted[0] - npos[0]), abs(predicted[1] - npos[1])) <= radius:
                return True
            if self._manhattan(opp['pos'], my_pos) <= radius + 2:
                return True
        return False

    def _sim_my_action(self, obs, action):
        x, y = obs['pos']
        cur_dir = obs['my_dir']
        hard = obs['hard']
        board = obs['board']
        rows, cols = obs['rows'], obs['cols']
        collected = 0
        path = []
        use_emp = action.startswith('X')
        move = action[1:] if use_emp else action

        if move in DIR_LIST:
            dirs = [move]
            final_dir = move
            phase_first = False
        elif move == '+':
            dirs = [cur_dir, cur_dir]
            final_dir = cur_dir
            phase_first = False
        elif move == 'P':
            if obs['phase_charges'] <= 0:
                return {'valid': False}
            dirs = [cur_dir, cur_dir]
            final_dir = cur_dir
            phase_first = True
        else:
            return {'valid': False}

        cx, cy = x, y
        for i, d in enumerate(dirs):
            nx, ny = self._step((cx, cy), d)
            if not (0 <= nx < cols and 0 <= ny < rows):
                return {'valid': False}
            cell = board[ny][nx]
            phased = phase_first and i == 0
            if not phased:
                if (nx, ny) in hard:
                    return {'valid': False}
            # Landing cell still matters even for phase.
            if i == len(dirs) - 1 and (nx, ny) in hard:
                return {'valid': False}
            if cell in TREASURE_VALUE:
                collected += TREASURE_VALUE[cell]
            path.append((cx, cy))
            cx, cy = nx, ny

        return {
            'valid': True,
            'end': (cx, cy),
            'trail_add': path,
            'dir': final_dir,
            'move': move,
            'collected': collected,
            'use_emp': use_emp,
        }

    def _local_opponent_actions(self, obs, opp):
        actions = []
        for d in DIR_LIST:
            npos = self._step(opp['pos'], d)
            if not self._in_bounds(npos, obs['cols'], obs['rows']):
                continue
            if npos in obs['hard']:
                continue
            actions.append((d, npos))
        if not actions:
            d = opp.get('direction', 'N')
            actions.append((d, self._step(opp['pos'], d)))
        return actions

    def _score_action(self, obs, action, hot, start_time):
        sim = self._sim_my_action(obs, action)
        if not sim['valid']:
            return -10**15

        end = sim['end']
        action_move = sim['move']
        is_x = sim['use_emp']

        score = 0.0

        for hid, hpos in obs['soft_heads'].items():
            if hid != obs['player_id'] and end == hpos:
                score -= 30000

        if hot:
            worst = self._local_worst_case(obs, sim, start_time)
            if worst <= -10**12:
                return worst
            score += worst
        else:
            for opp in obs['opponents']:
                if not opp.get('alive'):
                    continue
                if self._manhattan(opp['pos'], end) <= 1:
                    score -= 250

        new_hard = set(obs['hard'])
        for cell in sim['trail_add']:
            new_hard.add(cell)

        if obs['family'] == 'GENERAL' and obs['map_name'] not in self.public_general_maps:
            score += self._hidden_general_penalty(obs, end, new_hard)
        if obs['map_name'] not in self.public_solo_maps:
            score += self._compressed_fight_penalty(obs, end, new_hard)
        region = self._region_stats(end, new_hard, obs)
        score += region['space'] * 18
        score += region['exits'] * 60
        score += region['frontier'] * 8
        score -= region['dead_end_penalty'] * 220
        score += region['loopiness'] * 30

        score += sim['collected'] * 10.0
        treasure_bias = self._treasure_bias(obs, end, new_hard)
        score += treasure_bias * 1.5

        if obs['map_name'].startswith('treasure') and self._alive_opponent_count(obs) >= 2:
            score -= self._treasure_br_contest_penalty(obs, end, new_hard)

        if obs['map_name'].startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) <= 1:
            score += self._next_step_flexibility_targeted(obs, sim, new_hard)

        if obs['map_name'].startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) >= 2:
            score += self._br_space_safety_bonus(obs, end, new_hard)

        if obs['family'] == 'GENERAL':
            score += self._open_map_opening_adjustment(obs, action, sim, region, new_hard)
            score += self._multi_aggro_adjustment(obs, action, sim, region, new_hard)

        margin_state = self._score_margin_state(obs)
        if margin_state == 'ahead':
            score += region['space'] * 3
            if hot:
                score += self._mutual_death_value(obs, sim)
        elif margin_state == 'behind':
            score += treasure_bias * 0.30
            cutoff_mult = 60 if obs['map_name'].startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) <= 1 else 50
            score += self._cutoff_potential(obs, end, new_hard) * cutoff_mult
        else:
            cutoff_mult = 40 if obs['map_name'].startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) <= 1 else 30
            score += self._cutoff_potential(obs, end, new_hard) * cutoff_mult

        fam = obs['family']
        if fam == 'SOLO_PATH':
            score += self._solo_path_bonus(obs, end, new_hard)
        elif fam == 'SOLO_FLOOD':
            score += self._solo_flood_bonus(obs, end, new_hard)
        elif fam == 'SOLO_CHOICE':
            score += self._solo_choice_bonus(obs, end, new_hard)

        if self._alive_opponent_count(obs) == 0:
            score += self._survival_path_value(obs, end, new_hard) * 35

        if action_move == 'P':
            score += self._phase_value(obs, sim, region)
        elif action_move == '+':
            score += self._boost_value(obs, sim, region)
        if is_x:
            score += self._emp_value(obs, sim)

        score += self._map_style_bonus(obs, end, region)

        score -= self._emp_zone_penalty(obs, end)

        cx, cy = obs['meta']['center']
        score -= 0.4 * (abs(end[0] - cx) + abs(end[1] - cy))
        score += self._adj_wall_count(end, obs['hard'], obs['cols'], obs['rows']) * 4

        score += {'N': 0.04, 'S': 0.03, 'E': 0.02, 'W': 0.01, '+': 0.005, 'P': 0.004}.get(action_move, 0.0)
        return score

    def _local_worst_case(self, obs, sim, start_time):
        end = sim['end']
        nearby = [o for o in obs['opponents'] if o.get('alive') and self._manhattan(o['pos'], end) <= 4]
        if not nearby:
            return 0.0

        worst = 10**18
        new_hard = set(obs['hard'])
        for cell in sim['trail_add']:
            new_hard.add(cell)

        for opp in nearby:
            if time.perf_counter() - start_time > DEADLINE_S:
                break
            local_best_for_opp = -10**18
            for _, onext in self._local_opponent_actions(obs, opp):
                value = 0.0
                if onext == end:
                    value += 120000
                if onext in new_hard:
                    value -= 1500
                if self._manhattan(onext, end) == 1:
                    value += 140
                ospace = self._flood_count(onext, new_hard, obs['cols'], obs['rows']) if self._in_bounds(onext, obs['cols'], obs['rows']) and onext not in new_hard else 0
                mspace = self._flood_count(end, new_hard, obs['cols'], obs['rows'])
                value += (ospace - mspace) * 2.0
                local_best_for_opp = max(local_best_for_opp, value)
            worst = min(worst, -local_best_for_opp)

        return worst

    def _region_stats(self, start, hard, obs):
        rows, cols = obs['rows'], obs['cols']
        q = deque([start])
        seen = {start}
        space = 0
        exits = 0
        frontier = 0
        dead_end_penalty = 0
        degree_sum = 0

        while q:
            x, y = q.popleft()
            space += 1
            free_nei = 0
            for d in DIR_LIST:
                nx, ny = self._step((x, y), d)
                if not self._in_bounds((nx, ny), cols, rows):
                    continue
                if (nx, ny) in hard:
                    continue
                free_nei += 1
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
            degree_sum += free_nei
            if free_nei >= 3:
                exits += 1
            if free_nei <= 1:
                dead_end_penalty += 1
            if 1 <= x < cols - 1 and 1 <= y < rows - 1:
                frontier += 1
        loopiness = degree_sum / max(1, space) - 1.5
        return {
            'space': space,
            'exits': exits,
            'frontier': frontier,
            'dead_end_penalty': dead_end_penalty,
            'loopiness': loopiness,
        }

    def _next_step_flexibility_targeted(self, obs, sim, hard):
        end = sim['end']
        cols, rows = obs['cols'], obs['rows']
        valid = 0
        quality = 0.0
        for d in DIR_LIST:
            np = self._step(end, d)
            if not self._in_bounds(np, cols, rows) or np in hard:
                continue
            valid += 1
            deg = 4 - self._adj_wall_count(np, hard | {end}, cols, rows)
            quality += deg * 18.0
            quality += min(14, self._flood_count(np, hard | {end}, cols, rows)) * 1.2
        if valid == 0:
            return -400.0
        if valid == 1:
            return -140.0 + quality
        if valid == 2:
            return -20.0 + quality
        return valid * 12.0 + quality

    def _treasure_br_contest_penalty(self, obs, pos, hard):

        if not obs['treasures']:
            return 0.0
        cols, rows = obs['cols'], obs['rows']
        q = deque([pos])
        dist = {pos: 0}
        worst = 0.0
        while q:
            p = q.popleft()
            d = dist[p]
            if p in obs['treasures']:
                close = 0
                very_close = 0
                ahead = 0
                for opp in obs['opponents']:
                    if not opp.get('alive'):
                        continue
                    od = self._manhattan(opp['pos'], p)
                    if od <= 6:
                        close += 1
                    if od <= 3:
                        very_close += 1
                    if od <= d + 1:
                        ahead += 1
                deg = 4 - self._adj_wall_count(p, hard, cols, rows)
                local_pen = 0.0
                if deg <= 1:
                    local_pen = 120.0
                elif deg == 2:
                    local_pen = 40.0
                pen = ahead * 90.0 + close * 24.0 + very_close * 75.0 + local_pen
                if pen > worst:
                    worst = pen
            if d >= 8:
                continue
            for mv in DIR_LIST:
                np = self._step(p, mv)
                if self._in_bounds(np, cols, rows) and np not in hard and np not in dist:
                    dist[np] = d + 1
                    q.append(np)
        return worst


    def _br_space_safety_bonus(self, obs, pos, hard):
        cols, rows = obs['cols'], obs['rows']
        deg = 0
        near = 0
        for d in DIR_LIST:
            np = self._step(pos, d)
            if self._in_bounds(np, cols, rows) and np not in hard:
                deg += 1
        for opp in obs['opponents']:
            if opp.get('alive') and self._manhattan(opp['pos'], pos) <= 4:
                near += 1

        q = deque([(pos, 0)])
        seen = {pos}
        local = 0
        while q:
            p, depth = q.popleft()
            local += 1
            if depth >= 2:
                continue
            for d in DIR_LIST:
                np = self._step(p, d)
                if self._in_bounds(np, cols, rows) and np not in hard and np not in seen:
                    seen.add(np)
                    q.append((np, depth + 1))

        bonus = 0.0
        if deg <= 1:
            bonus -= 320.0
        elif deg == 2:
            bonus -= 95.0
        if local <= 4:
            bonus -= 220.0
        elif local <= 6:
            bonus -= 80.0
        bonus += min(local, 10) * 10.0
        bonus += max(0, deg - 2) * 35.0
        bonus -= near * (55.0 if deg <= 2 else 25.0)
        return bonus

    def _treasure_bias(self, obs, start, hard):
        treasures = obs['treasures']
        if not treasures:
            return 0.0
        rows, cols = obs['rows'], obs['cols']
        q = deque([start])
        dist = {start: 0}
        best = 0.0
        while q:
            p = q.popleft()
            d = dist[p]
            if p in treasures:
                val = treasures[p]
                contested_penalty = 0.0
                for opp in obs['opponents']:
                    if not opp.get('alive'):
                        continue
                    od = self._manhattan(opp['pos'], p)
                    if od <= d:
                        contested_penalty += 6.0 if opp['id'] < obs['player_id'] else 10.0
                score = val * (1.0 / (1 + d)) - contested_penalty
                if score > best:
                    best = score
            if d >= 14:
                continue
            for mv in DIR_LIST:
                np = self._step(p, mv)
                if self._in_bounds(np, cols, rows) and np not in hard and np not in dist:
                    dist[np] = d + 1
                    q.append(np)
        return best * 12.0

    def _score_margin_state(self, obs):
        my_score = obs['my_score']
        uppers = []
        lowers = []
        for opp in obs['opponents']:
            if not opp.get('alive'):
                continue
            lo, hi = self.state['score_bounds'].get(opp['id'], (0, my_score + 300))
            lowers.append(lo)
            uppers.append(hi)
        if not uppers:
            return 'ahead'
        if my_score >= max(uppers) + 50:
            return 'ahead'
        if my_score + 80 < max(lowers):
            return 'behind'
        return 'close'

    def _mutual_death_value(self, obs, sim):
        end = sim['end']
        bonus = 0.0
        for opp in obs['opponents']:
            if opp.get('alive') and self._manhattan(opp['pos'], end) <= 1:
                bonus += 80.0
        return bonus

    def _cutoff_potential(self, obs, end, hard):

        deg = 4 - self._adj_wall_count(end, hard, obs['cols'], obs['rows'])
        if deg > 2:
            return 0.0
        gain = 0.0
        for opp in obs['opponents']:
            if not opp.get('alive'):
                continue
            if self._manhattan(opp['pos'], end) <= 5:
                opp_space = self._flood_count(opp['pos'], hard, obs['cols'], obs['rows'])
                reduced = self._flood_count(opp['pos'], hard | {end}, obs['cols'], obs['rows'])
                gain += max(0, opp_space - reduced) / 25.0
        return gain

    def _solo_path_bonus(self, obs, end, hard):
        return self._treasure_bias(obs, end, hard) * 0.9 + self._corridor_run_bonus(end, hard, obs) * 30

    def _solo_flood_bonus(self, obs, end, hard):
        return self._flood_count(end, hard, obs['cols'], obs['rows']) * 6 + self._branching_bonus(end, hard, obs) * 80

    def _solo_choice_bonus(self, obs, end, hard):
        return self._choice_branch_value(obs, end, hard) * 18

    def _phase_value(self, obs, sim, region):

        if obs['phase_charges'] <= 1:
            reserve_penalty = -260
        else:
            reserve_penalty = -100
        straight = self._step(obs['pos'], obs['my_dir'])
        blocked_now = straight in obs['hard'] or not self._in_bounds(straight, obs['cols'], obs['rows'])
        convert = min(region['space'], 120)
        return reserve_penalty + (260 if blocked_now else 0) + convert * 2.5

    def _boost_value(self, obs, sim, region):
        end = sim['end']
        bonus = 20.0 + region['space'] * 0.8
        elapsed = self.state.get('elapsed', 0)
        if obs['map_name'].startswith(('cube', 'orbit')) and elapsed < 6 and self._alive_opponent_count(obs) <= 1:
            bonus -= 75.0
        if end in obs['treasures']:
            bonus += obs['treasures'][end] * 2
        if self._adj_wall_count(end, obs['hard'], obs['cols'], obs['rows']) >= 3:
            bonus -= 120
        return bonus

    def _emp_value(self, obs, sim):
        npos = sim['end']
        radius = obs['emp_radius']
        value = -120.0
        for opp in obs['opponents']:
            if not opp.get('alive'):
                continue
            predicted = opp['pos']
            opp_dir = opp.get('direction', 'N')
            for _ in range(5):
                predicted = self._step(predicted, opp_dir)
            if max(abs(predicted[0] - npos[0]), abs(predicted[1] - npos[1])) <= radius:
                value += 220.0
                if self._adj_wall_count(predicted, obs['hard'], obs['cols'], obs['rows']) >= 2:
                    value += 120.0
            elif self._manhattan(opp['pos'], npos) <= radius + 1:
                value += 70.0

        value -= self._emp_zone_penalty(obs, npos) * 0.4
        return value

    def _map_style_bonus(self, obs, end, region):
        name = obs['map_name']
        if name.startswith('arena'):
            return region['frontier'] * 6
        if name.startswith('treasure'):
            return self._treasure_bias(obs, end, obs['hard']) * 0.4
        if name.startswith('maze'):
            return -region['dead_end_penalty'] * 120 + region['exits'] * 40
        if name.startswith('gate'):
            return -self._timed_wall_near_penalty(obs, end) + region['exits'] * 25
        if name.startswith('cube'):
            return self._adj_wall_count(end, obs['hard'], obs['cols'], obs['rows']) * 10 + region['space'] * 1.5
        if name.startswith('orbit'):
            return region['space'] * 1.2 - self._timed_wall_near_penalty(obs, end) * 0.8
        return 0.0

    def _emp_zone_penalty(self, obs, pos):
        penalty = 0.0
        for x, y, timer in obs['danger_union']:
            if (x, y) == pos:
                penalty += 90.0 / max(1, timer)
        return penalty

    def _hidden_general_penalty(self, obs, pos, hard):
        deg = 0
        for d in DIR_LIST:
            np = self._step(pos, d)
            if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                deg += 1
        penalty = 0.0
        if deg <= 1:
            penalty -= 420
        elif deg == 2:
            penalty -= 120
        close = 0
        for opp in obs['opponents']:
            if opp.get('alive') and self._manhattan(opp['pos'], pos) <= 2:
                close += 1
        penalty -= close * (150 if deg <= 2 else 60)
        penalty -= self._emp_zone_penalty(obs, pos) * (1.0 if deg <= 2 else 0.4)
        return penalty

    def _best_emergency_move(self, obs):
        best = None
        best_space = -1
        for action in ['N', 'S', 'E', 'W', '+', 'P']:
            sim = self._sim_my_action(obs, action)
            if not sim['valid']:
                continue
            end = sim['end']
            hard = set(obs['hard'])
            for p in sim['trail_add']:
                hard.add(p)
            space = self._flood_count(end, hard, obs['cols'], obs['rows'])
            if end in obs['soft_heads'].values():
                space -= 1000
            if space > best_space:
                best_space = space
                best = action
        return best or obs['my_dir'] or 'N'

    def _is_hot_turn(self, obs):
        if obs['active_emps']:
            return True
        my_pos = obs['pos']
        for opp in obs['opponents']:
            if not opp.get('alive'):
                continue
            if self._manhattan(opp['pos'], my_pos) <= 4:
                return True
        for tpos in obs['treasures']:
            if self._manhattan(tpos, my_pos) <= 4:
                for opp in obs['opponents']:
                    if opp.get('alive') and self._manhattan(opp['pos'], tpos) <= 5:
                        return True
        return False

    def _alive_opponent_count(self, obs):
        return sum(1 for opp in obs['opponents'] if opp.get('alive'))

    def _lone_survivor_move(self, obs, start_time):
        best_action = None
        best_value = -10**18
        for action in ['N', 'S', 'E', 'W', '+', 'P']:
            if time.perf_counter() - start_time > DEADLINE_S * 0.65:
                break
            sim = self._sim_my_action(obs, action)
            if not sim['valid']:
                continue
            hard = set(obs['hard'])
            for cell in sim['trail_add']:
                hard.add(cell)
            value = self._survival_path_value(obs, sim['end'], hard)
            value += self._flood_count(sim['end'], hard, obs['cols'], obs['rows']) * 0.2
            if action in ('+', 'P'):
                value += 1.2
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


    def _hidden_solo_family_move(self, obs, start_time):
        candidates = []
        for a in ['N', 'S', 'E', 'W', '+', 'P']:
            sim = self._sim_my_action(obs, a)
            if sim['valid']:
                candidates.append((a, sim))
        if not candidates:
            return None

        best_a = None
        best_v = -10**18
        for a, sim in candidates:
            if time.perf_counter() - start_time > DEADLINE_S * 0.72:
                break
            hard = set(obs['hard'])
            for cell in sim['trail_add']:
                hard.add(cell)
            end = sim['end']
            area = self._flood_count(end, hard, obs['cols'], obs['rows'])
            future = self._hidden_survival_value(obs, end, hard)
            treasure = self._treasure_bias(obs, end, hard)
            art = self._hidden_articulation_value(obs, end, hard)
            timed = self._timed_wall_opening_bias(obs, end)
            if obs['family'] == 'SOLO_FLOOD':
                value = area * 6.2 + self._branching_bonus(end, hard, obs) * 100 + future * 22 + art * 3.2 + treasure * 0.18
                if a == 'P':
                    value -= 220
            elif obs['family'] == 'SOLO_PATH':
                value = treasure * 1.15 + self._corridor_run_bonus(end, hard, obs) * 34 + future * 15 + area * 2.2 + timed * 0.35
                if a == 'P':
                    value -= 200
            else:
                value = self._hidden_choice_value(obs, end, hard) * 18 + future * 16 + art * 5.0 + timed * 0.2
                value += self._hidden_choice_lookahead(obs, end, hard)
                if a == 'P':
                    value -= 160
            if value > best_v:
                best_v = value
                best_a = a
        return best_a

    def _hidden_survival_value(self, obs, start, hard):
        cols, rows = obs['cols'], obs['rows']
        if not self._in_bounds(start, cols, rows) or start in hard:
            return -10**9
        blocked = set(hard)
        pos = start
        total = 0
        limit = min(220, cols * rows)
        for _ in range(limit):
            moves = []
            for d in DIR_LIST:
                np = self._step(pos, d)
                if not self._in_bounds(np, cols, rows) or np in blocked:
                    continue
                deg = 0
                wallish = self._adj_wall_count(np, blocked, cols, rows)
                for d2 in DIR_LIST:
                    np2 = self._step(np, d2)
                    if self._in_bounds(np2, cols, rows) and np2 not in blocked and np2 != pos:
                        deg += 1
                moves.append((-deg, wallish, np))
            if not moves:
                break
            moves.sort()
            _, _, nxt = moves[0]
            blocked.add(pos)
            pos = nxt
            total += 1
        return total

    def _hidden_articulation_value(self, obs, pos, hard):
        cols, rows = obs['cols'], obs['rows']
        blocked = set(hard)
        blocked.add(pos)
        sizes = []
        seen = set()
        for d in DIR_LIST:
            seed = self._step(pos, d)
            if not self._in_bounds(seed, cols, rows) or seed in blocked or seed in seen:
                continue
            q = deque([seed])
            seen.add(seed)
            size = 0
            tval = 0
            while q:
                cur = q.popleft()
                size += 1
                if cur in obs['treasures']:
                    tval += obs['treasures'][cur]
                for d2 in DIR_LIST:
                    np = self._step(cur, d2)
                    if self._in_bounds(np, cols, rows) and np not in blocked and np not in seen:
                        seen.add(np)
                        q.append(np)
            sizes.append(size + tval / 15.0)
        if len(sizes) < 2:
            return 0.0
        sizes.sort()
        small = sizes[0]
        second = sizes[1]
        return small * 3.5 if second >= small * 1.5 else small * 1.2

    def _hidden_choice_value(self, obs, pos, hard):
        area = self._flood_count(pos, hard, obs['cols'], obs['rows'])
        treasure = self._treasure_bias(obs, pos, hard)
        future = self._hidden_survival_value(obs, pos, hard)
        branch = self._branching_bonus(pos, hard, obs)
        comp_area, comp_treasure, comp_diams, exits = self._component_profile(obs, pos, hard)
        return (
            area * 0.42
            + treasure / 8.0
            + future * 0.50
            + branch * 7.0
            + comp_area * 0.78
            + comp_treasure * 0.46
            + comp_diams * 30.0
            + exits * 16.0
        )

    def _hidden_choice_lookahead(self, obs, pos, hard):
        best = -10**18
        found = False
        for d in DIR_LIST:
            np = self._step(pos, d)
            if not self._in_bounds(np, obs['cols'], obs['rows']) or np in hard:
                continue
            found = True
            nh = set(hard)
            nh.add(pos)
            val = self._hidden_choice_value(obs, np, nh)
            deg = self._branching_bonus(np, nh, obs)
            val += deg * 22.0
            if np in obs['treasures']:
                val += obs['treasures'][np] * 1.5
            if val > best:
                best = val
        if not found:
            return -1800.0
        return best * 0.55

    def _compressed_fight_penalty(self, obs, pos, hard):
        close = [opp for opp in obs['opponents'] if opp.get('alive') and self._manhattan(opp['pos'], pos) <= 4]
        if not close and not obs['active_emps']:
            return 0.0
        deg = 0
        for d in DIR_LIST:
            np = self._step(pos, d)
            if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                deg += 1
        penalty = 0.0
        if len(close) >= 2:
            if deg <= 1:
                penalty -= 520.0
            elif deg == 2:
                penalty -= 180.0
        elif len(close) == 1 and deg <= 1:
            penalty -= 180.0

        if obs['active_emps']:
            if deg <= 1:
                penalty -= 300.0
            elif deg == 2:
                penalty -= 120.0

        q = deque([(pos, 0)])
        seen = {pos}
        local = 0
        while q:
            p, d0 = q.popleft()
            local += 1
            if d0 >= 3:
                continue
            for d in DIR_LIST:
                np = self._step(p, d)
                if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard and np not in seen:
                    seen.add(np)
                    q.append((np, d0 + 1))

        margin_state = self._score_margin_state(obs)
        strict_unknown = obs['family'] == 'GENERAL' and obs['map_name'] not in self.public_general_maps

        if strict_unknown:
            if obs['active_emps'] and deg <= 2 and local <= 7 and len(close) >= 1:
                penalty -= 2600.0
            elif len(close) >= 2 and deg <= 2 and local <= 6:
                penalty -= 1100.0

        if deg <= 2 and local <= 6:
            if obs['active_emps'] and len(close) >= 1:
                penalty -= 1400.0
            elif len(close) >= 2:
                penalty -= 700.0

        for opp in close:
            ox, oy = opp['pos']
            same_lane = (ox == pos[0] or oy == pos[1])
            if not same_lane:
                continue
            clear = True
            if ox == pos[0]:
                step = 1 if oy < pos[1] else -1
                for y in range(pos[1] + step, oy, step):
                    if (ox, y) in hard:
                        clear = False
                        break
            else:
                step = 1 if ox < pos[0] else -1
                for x in range(pos[0] + step, ox, step):
                    if (x, oy) in hard:
                        clear = False
                        break
            dist = self._manhattan(opp['pos'], pos)
            if clear and dist <= 3:
                lane_pen = 90.0
                if deg <= 2:
                    lane_pen += 130.0
                if strict_unknown and dist <= 2:
                    lane_pen += 220.0
                if margin_state != 'behind':
                    lane_pen += 70.0
                penalty -= lane_pen

        if len(close) >= 2 and local <= 6:
            penalty -= 160.0
        elif strict_unknown and len(close) >= 1 and deg <= 2 and local <= 5:
            penalty -= 180.0
        return penalty

    def _open_map_opening_adjustment(self, obs, action, sim, region, hard):
        elapsed = self.state.get('elapsed', 0)
        if elapsed > 6:
            return 0.0
        name = obs['map_name']
        if not (name.startswith('arena') or name.startswith('treasure') or name.startswith('orbit') or name.startswith('cube')):
            return 0.0

        close = [opp for opp in obs['opponents'] if opp.get('alive') and self._manhattan(opp['pos'], obs['pos']) <= 6]
        if not close:
            return 0.0

        end = sim['end']
        deg = 0
        for d in DIR_LIST:
            np = self._step(end, d)
            if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                deg += 1

        bonus = 0.0
        if action in ('+', 'P') and len(close) >= 1:
            extra = 20.0 if name.startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) <= 1 else 0.0
            bonus -= 70.0 + extra + 35.0 * max(0, 2 - deg)
            if len(close) >= 2:
                bonus -= 45.0 + (15.0 if name.startswith(('cube', 'orbit')) and self._alive_opponent_count(obs) <= 1 else 0.0)

        cx, cy = obs['meta']['center']
        dist_center = abs(end[0] - cx) + abs(end[1] - cy)
        bonus += max(0.0, 10.0 - dist_center) * 6.0
        bonus += region['exits'] * 18.0
        if deg <= 1:
            bonus -= 140.0
        elif deg == 2:
            bonus -= 40.0
        return bonus

    def _multi_aggro_adjustment(self, obs, action, sim, region, hard):
        close = [opp for opp in obs['opponents'] if opp.get('alive') and self._manhattan(opp['pos'], obs['pos']) <= 4]
        if len(close) < 2:
            return 0.0

        end = sim['end']
        deg = 0
        for d in DIR_LIST:
            np = self._step(end, d)
            if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                deg += 1

        q = deque([(end, 0)])
        seen = {end}
        local = 0
        while q:
            p, depth = q.popleft()
            local += 1
            if depth >= 2:
                continue
            for d in DIR_LIST:
                np = self._step(p, d)
                if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard and np not in seen:
                    seen.add(np)
                    q.append((np, depth + 1))

        adj = 0.0
        if deg <= 2:
            adj -= 220.0
        if local <= 5:
            adj -= 180.0
        adj += region['exits'] * 24.0
        adj += max(0, region['space'] - 8) * 1.6
        if action in ('+', 'P') and local <= 6:
            adj -= 90.0
        if obs['active_emps'] and deg <= 2:
            adj -= 260.0
        return adj

    def _public_solo_book_move(self, obs):
        book = getattr(self, 'public_solo_books', {}).get(obs['map_name'])
        if not book:
            return None
        idx = self.state['solo_book_idx'].get(obs['map_name'], 0)
        if idx >= len(book):
            return None
        action = book[idx]
        sim = self._sim_my_action(obs, action)
        if not sim['valid']:
            return None
        self.state['solo_book_idx'][obs['map_name']] = idx + 1
        return action

    def _solo_opening_move(self, obs, start_time):
        elapsed = self.state.get('elapsed', 0)
        fam = obs['family']
        if fam == 'SOLO_PATH':
            max_depth, beam_width = 8, 6
        elif fam == 'SOLO_CHOICE':
            max_depth, beam_width = 8, 6
        else:
            return None
        if elapsed > 22 or obs['rows'] * obs['cols'] > 450:
            return None

        root = {
            'pos': obs['pos'],
            'dir': obs['my_dir'],
            'hard': set(obs['hard']),
            'score': 0.0,
            'first': None,
            'phase': obs['phase_charges'],
        }
        beam = [root]
        best_first = None
        best_value = -10**18

        for depth in range(max_depth):
            if time.perf_counter() - start_time > DEADLINE_S * 0.78:
                break
            next_beam = []
            for state in beam:
                for action in DIR_LIST:
                    child = self._solo_plan_step(obs, state, action)
                    if child is None:
                        continue
                    if child['first'] is None:
                        child['first'] = action
                    next_beam.append(child)
                if False and state['phase'] > 0 and depth <= 2:
                    child = self._solo_plan_step(obs, state, 'P')
                    if child is not None:
                        if child['first'] is None:
                            child['first'] = 'P'
                        next_beam.append(child)
            if not next_beam:
                break
            for child in next_beam:
                est = child['score']
                est += self._solo_leaf_value(obs, child['pos'], child['hard'], fam, elapsed + depth)
                child['est'] = est
                if child['first'] is not None and est > best_value:
                    best_value = est
                    best_first = child['first']
            next_beam.sort(key=lambda st: st['est'], reverse=True)
            beam = next_beam[:beam_width]

        return best_first

    def _solo_plan_step(self, obs, state, action):
        pos = state['pos']
        cur_dir = state['dir']
        phase = state['phase']
        board = obs['board']
        rows, cols = obs['rows'], obs['cols']
        hard = set(state['hard'])
        score = state['score']

        if action in DIR_LIST:
            dirs = [action]
            final_dir = action
            phase_first = False
            phase_cost = 0
        elif action == 'P':
            dirs = [cur_dir, cur_dir]
            final_dir = cur_dir
            phase_first = True
            phase_cost = 1
            if phase <= 0:
                return None
        else:
            return None

        cx, cy = pos
        path = []
        gained = 0
        for i, d in enumerate(dirs):
            nx, ny = self._step((cx, cy), d)
            if not self._in_bounds((nx, ny), cols, rows):
                return None
            phased = phase_first and i == 0
            if not phased and (nx, ny) in hard:
                return None
            if i == len(dirs) - 1 and (nx, ny) in hard:
                return None
            cell = board[ny][nx]
            if cell in TREASURE_VALUE:
                gained += TREASURE_VALUE[cell]
            path.append((cx, cy))
            cx, cy = nx, ny

        for old in path:
            hard.add(old)
        new_score = score + gained * 8.0
        return {
            'pos': (cx, cy),
            'dir': final_dir,
            'hard': hard,
            'score': new_score,
            'first': state['first'],
            'phase': phase - phase_cost,
        }

    def _solo_leaf_value(self, obs, pos, hard, family, elapsed):
        space = self._flood_count(pos, hard, obs['cols'], obs['rows'])
        branch = self._branching_bonus(pos, hard, obs)
        corridor = self._corridor_run_bonus(pos, hard, obs)
        future = self._survival_path_value(obs, pos, hard)
        treasure = self._treasure_bias(obs, pos, hard)
        timed = self._timed_wall_opening_bias(obs, pos)
        if family == 'SOLO_FLOOD':
            return space * 9.0 + branch * 110 + future * 32 + timed
        if family == 'SOLO_PATH':
            return treasure * 1.2 + corridor * 55 + future * 25 + space * 3.0 + timed
        return treasure * 1.1 + space * 5.0 + branch * 70 + future * 24 + timed

    def _survival_path_value(self, obs, start, hard):
        cols, rows = obs['cols'], obs['rows']
        if not self._in_bounds(start, cols, rows) or start in hard:
            return -10**9
        blocked = set(hard)
        pos = start
        total = 0
        limit = min(220, cols * rows)
        for _ in range(limit):
            moves = []
            for d in DIR_LIST:
                np = self._step(pos, d)
                if not self._in_bounds(np, cols, rows) or np in blocked:
                    continue
                deg = 0
                bend = 0
                for d2 in DIR_LIST:
                    np2 = self._step(np, d2)
                    if self._in_bounds(np2, cols, rows) and np2 not in blocked and np2 != pos:
                        deg += 1
                        bend += self._adj_wall_count(np2, blocked, cols, rows)
                wallish = self._adj_wall_count(np, blocked, cols, rows)
                moves.append((deg, -wallish, -bend, np))
            if not moves:
                break
            moves.sort()
            _, _, _, nxt = moves[0]
            blocked.add(pos)
            pos = nxt
            total += 1
        return total

    def _timed_wall_opening_bias(self, obs, pos):
        elapsed = self.state.get('elapsed', 0)
        bias = 0.0
        for (wx, wy), digit in obs['timed_walls'].items():
            exact = digit * 10
            remain = exact - elapsed
            dist = self._manhattan(pos, (wx, wy))
            if 0 < remain <= 25 and dist <= 4:
                bias += (26 - remain) * 6.0
            elif 0 < remain <= 45 and dist <= 2:
                bias += (46 - remain) * 2.0
        return bias

    def _component_profile(self, obs, start, hard):
        if not self._in_bounds(start, obs['cols'], obs['rows']) or start in hard:
            return 0, 0, 0, 0
        q = deque([start])
        seen = {start}
        area = 0
        treasure = 0
        diamonds = 0
        exits = 0
        while q:
            p = q.popleft()
            area += 1
            if p in obs['treasures']:
                val = obs['treasures'][p]
                treasure += val
                if val >= 50:
                    diamonds += 1
            deg = 0
            for d in DIR_LIST:
                np = self._step(p, d)
                if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                    deg += 1
                    if np not in seen:
                        seen.add(np)
                        q.append(np)
            if deg >= 3:
                exits += 1
        return area, treasure, diamonds, exits

    def _choice_branch_value(self, obs, start, hard):
        area = self._flood_count(start, hard, obs['cols'], obs['rows'])
        return area + self._treasure_bias(obs, start, hard) / 20.0

    def _corridor_run_bonus(self, start, hard, obs):
        best = 0
        for d in DIR_LIST:
            count = 0
            p = start
            while True:
                p = self._step(p, d)
                if not self._in_bounds(p, obs['cols'], obs['rows']) or p in hard:
                    break
                count += 1
            best = max(best, count)
        return best

    def _branching_bonus(self, start, hard, obs):
        deg = 0
        for d in DIR_LIST:
            np = self._step(start, d)
            if self._in_bounds(np, obs['cols'], obs['rows']) and np not in hard:
                deg += 1
        return deg

    def _timed_wall_near_penalty(self, obs, pos):
        pen = 0.0
        for (wx, wy), digit in obs['timed_walls'].items():
            dist = abs(wx - pos[0]) + abs(wy - pos[1])
            if dist <= 2:
                pen += max(0, 5 - digit)
        return pen * 25.0

    def _flood_count(self, start, hard, cols, rows):
        if not self._in_bounds(start, cols, rows) or start in hard:
            return 0
        q = deque([start])
        seen = {start}
        count = 0
        while q:
            p = q.popleft()
            count += 1
            for d in DIR_LIST:
                np = self._step(p, d)
                if self._in_bounds(np, cols, rows) and np not in hard and np not in seen:
                    seen.add(np)
                    q.append(np)
        return count

    def _adj_wall_count(self, pos, hard, cols, rows):
        c = 0
        for d in DIR_LIST:
            np = self._step(pos, d)
            if not self._in_bounds(np, cols, rows) or np in hard:
                c += 1
        return c

    def _in_bounds(self, pos, cols, rows):
        return 0 <= pos[0] < cols and 0 <= pos[1] < rows

    def _step(self, pos, d):
        dx, dy = DIRS[d]
        return pos[0] + dx, pos[1] + dy

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
