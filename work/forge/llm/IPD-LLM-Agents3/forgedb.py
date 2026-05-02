#!/usr/bin/env python3
"""
    FORGE Database ETL — 3-Agent Version (ipd3 schema)
    ETL Pipeline for 3-Agent Iterated Prisoner's Dilemma with LLM Agents

    Changes from IPD-LLM-Agents2 forgedb.py:
      - Uses ipd3 schema instead of ipd2.
      - load_json extracts host_2 / model_2 from config for the third agent.
      - SQL views (experiment_summary_vw, episode_summary_vw, rounds_summary_vw)
        updated to include agent_2 columns.
      - ipd2.results table unchanged — schema is still agent-count agnostic for
        main tables (llm_agents, episodes, rounds loop dynamically).
      - ipd3.experiment_summary_vw and ipd3.episode_summary_vw now JOIN on
        agent_idx 0, 1, and 2 explicitly (because SQL pivoting requires explicit
        column definitions).

    Practicum I - MSDS 692/S41: Data Science Practicum I
    Emily D. Carpenter
    Anderson College of Business and Computing, Regis University
    Dr. Douglas Hart, Dr. Kellen Sorauf
    March 2026
"""

import argparse
import glob
import json
import logging
import os

import pandas as pd
import psycopg
from psycopg.rows import dict_row

script_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    filename=os.path.join(script_dir, 'forgedb3.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ForgeDB:
    def __init__(self, host='platinum', dbname='forge', user=None):
        if user is None:
            import getpass
            user = getpass.getuser()
        self.conn = psycopg.connect(
            host=host,
            dbname=dbname,
            user=user,
            row_factory=dict_row
        )

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------
    # Query methods (mirror ipd2 interface but target ipd3 schema)
    # ------------------------------------------------------------------
    def query(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def get_raw_data(self, **kwargs):
        return self._query_view('raw_data_vw', **kwargs)

    def get_results(self, **kwargs):
        return self._query_view('results_vw', **kwargs)

    def get_summary(self, **kwargs):
        return self._query_view('experiment_summary_vw', **kwargs)

    def get_episode_summary(self, **kwargs):
        return self._query_view('episode_summary_vw', **kwargs)

    def get_rounds_summary(self, **kwargs):
        return self._query_view('rounds_summary_vw', **kwargs)

    def get_rounds_detail(self, **kwargs):
        return self._query_view('rounds_detail_vw', **kwargs)

    def _query_view(self, view_name, start_date=None, end_date=None,
                    username=None, filename=None, comment=None, limit=None):
        try:
            sql = f"SELECT * FROM ipd3.{view_name} WHERE 1=1"
            params = {}

            if start_date is not None:
                sql += " AND timestamp >= %(start_date)s"
                params['start_date'] = start_date
            if end_date is not None:
                sql += " AND timestamp < %(end_date)s"
                params['end_date'] = end_date
            if username is not None:
                sql += " AND LOWER(username) LIKE LOWER(%(username)s)"
                params['username'] = username
            if filename is not None:
                sql += " AND LOWER(filename) LIKE LOWER(%(filename)s)"
                params['filename'] = filename
            if comment is not None:
                sql += " AND LOWER(comment) LIKE LOWER(%(comment)s)"
                params['comment'] = comment
            if limit is not None:
                sql += f" LIMIT {limit}"

            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            return pd.DataFrame(rows)

        except Exception as e:
            err_msg = f"_query_view({view_name}) failed - {e}"
            logging.error(err_msg)
            print(err_msg)
            raise

    # ------------------------------------------------------------------
    # Research log
    # ------------------------------------------------------------------
    def add_log(self, remarks, username=None, subject='General',
                log_dttm=None, tags=None):
        try:
            if username is None:
                import getpass
                username = getpass.getuser()
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ipd3.research_log (
                        create_dttm, log_dttm, username, subject, remarks, tags
                    ) VALUES (
                        NOW(), COALESCE(%(log_dttm)s, NOW()),
                        %(username)s, %(subject)s, %(remarks)s, %(tags)s
                    ) RETURNING log_id
                """, {
                    'log_dttm': log_dttm,
                    'username': username,
                    'subject':  subject,
                    'remarks':  remarks,
                    'tags':     tags
                })
                log_id = cur.fetchone()['log_id']
            self.conn.commit()
            logging.info(f"Added log entry {log_id}: {subject}")
            print(f"Added log entry {log_id}: {subject}")
            return log_id
        except Exception as e:
            self.conn.rollback()
            err_msg = f"Failed to add log entry - {e}"
            logging.error(err_msg)
            print(err_msg)
            raise

    def get_log(self, username=None, subject=None, remarks=None,
                tags=None, start_date=None, end_date=None, limit=None):
        try:
            sql = "SELECT * FROM ipd3.research_log WHERE 1=1"
            params = {}
            if username is not None:
                sql += " AND LOWER(username) LIKE LOWER(%(username)s)"
                params['username'] = username
            if subject is not None:
                sql += " AND LOWER(subject) LIKE LOWER(%(subject)s)"
                params['subject'] = subject
            if remarks is not None:
                sql += " AND LOWER(remarks) LIKE LOWER(%(remarks)s)"
                params['remarks'] = remarks
            if tags is not None:
                if isinstance(tags, list):
                    sql += " AND tags @> %(tags)s"
                else:
                    sql += " AND %(tags)s = ANY(tags)"
                params['tags'] = tags
            if start_date is not None:
                sql += " AND log_dttm >= %(start_date)s"
                params['start_date'] = start_date
            if end_date is not None:
                sql += " AND log_dttm < %(end_date)s"
                params['end_date'] = end_date
            sql += " ORDER BY log_dttm"
            if limit is not None:
                sql += f" LIMIT {limit}"
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            return pd.DataFrame(rows)
        except Exception as e:
            err_msg = f"Failed to query log entries - {e}"
            logging.error(err_msg)
            print(err_msg)
            raise

    def delete_log(self, log_id):
        try:
            with self.conn.cursor() as cur:
                if isinstance(log_id, tuple):
                    cur.execute(
                        "DELETE FROM ipd3.research_log WHERE log_id BETWEEN %(start)s AND %(end)s",
                        {'start': log_id[0], 'end': log_id[1]}
                    )
                elif isinstance(log_id, list):
                    cur.execute(
                        "DELETE FROM ipd3.research_log WHERE log_id = ANY(%(ids)s)",
                        {'ids': log_id}
                    )
                else:
                    cur.execute(
                        "DELETE FROM ipd3.research_log WHERE log_id = %(log_id)s",
                        {'log_id': log_id}
                    )
                deleted = cur.rowcount
            self.conn.commit()
            if deleted:
                logging.info(f"Deleted {deleted} log entry(ies)")
                print(f"Deleted {deleted} log entry(ies)")
            else:
                print(f"No entries found matching {log_id}")
            return deleted
        except Exception as e:
            self.conn.rollback()
            err_msg = f"Failed to delete log entry {log_id} - {e}"
            logging.error(err_msg)
            print(err_msg)
            raise

    # ------------------------------------------------------------------
    # JSON import — updated for 3 agents
    # ------------------------------------------------------------------
    def load_json(self, filepath, user_name='unknown'):
        """
        Import a 3-agent JSON results file into the ipd3 schema.

        Changes vs. ipd2 forgedb.py:
          - Reads data['config']['model_2'] for the third agent.
          - The llm_agents / episodes / rounds loops are already dynamic
            (they iterate while f'agent_{idx}' in data) — no loop change needed.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            filename   = os.path.basename(filepath)
            researcher = data.get('username', user_name)

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ipd3.results (
                        filename, timestamp, hostname, username,
                        elapsed_seconds,
                        cfg_num_episodes, cfg_round_per_episode, cfg_total_rounds,
                        cfg_history_window_size, cfg_temperature,
                        cfg_reset_between_episodes, cfg_reflection_type,
                        cfg_decision_token_limit, cfg_reflection_token_limit,
                        cfg_http_timeout, cfg_force_decision_retries,
                        system_prompt, reflection_template, raw_json, comment
                    ) VALUES (
                        %(filename)s, %(timestamp)s, %(hostname)s, %(username)s,
                        %(elapsed_seconds)s,
                        %(num_episodes)s, %(rounds_per_episode)s, %(total_rounds)s,
                        %(history_window_size)s, %(temperature)s,
                        %(reset_between_episodes)s, %(reflection_type)s,
                        %(decision_token_limit)s, %(reflection_token_limit)s,
                        %(http_timeout)s, %(force_decision_retries)s,
                        %(system_prompt)s, %(reflection_template)s,
                        %(raw_json)s, %(comment)s
                    ) RETURNING results_id
                """, {
                    'filename':               filename,
                    'timestamp':              data['timestamp'],
                    'hostname':               data.get('hostname', None),
                    'username':               researcher,
                    'elapsed_seconds':        data['elapsed_seconds'],
                    'num_episodes':           data['config']['num_episodes'],
                    'rounds_per_episode':     data['config']['rounds_per_episode'],
                    'total_rounds':           data['config']['total_rounds'],
                    'history_window_size':    data['config']['history_window_size'],
                    'temperature':            data['config']['temperature'],
                    'reset_between_episodes': data['config']['reset_between_episodes'],
                    'reflection_type':        data['config']['reflection_type'],
                    'decision_token_limit':   data['config']['decision_token_limit'],
                    'reflection_token_limit': data['config']['reflection_token_limit'],
                    'http_timeout':           data['config']['http_timeout'],
                    'force_decision_retries': data['config']['force_decision_retries'],
                    'system_prompt':          data['prompts']['system_prompt'],
                    'reflection_template':    data['prompts']['reflection_template'],
                    'raw_json':               json.dumps(data),
                    'comment':                data.get('comment', None)
                })
                results_id = cur.fetchone()['results_id']

                # Insert agent rows (dynamic — handles 2 or 3 agents)
                agent_idx = 0
                while f'agent_{agent_idx}' in data:
                    agent_key = f'agent_{agent_idx}'
                    host_key  = f'host_{agent_idx}'
                    model_key = f'model_{agent_idx}'

                    cur.execute("""
                        INSERT INTO ipd3.llm_agents (
                            results_id, agent_idx, host, agent_model, cfg_model,
                            total_score, total_cooperations, overall_cooperation_rate
                        ) VALUES (
                            %(results_id)s, %(agent_idx)s, %(host)s,
                            %(agent_model)s, %(cfg_model)s,
                            %(total_score)s, %(total_cooperations)s,
                            %(overall_cooperation_rate)s
                        )
                    """, {
                        'results_id':              results_id,
                        'agent_idx':               agent_idx,
                        'host':                    data.get(host_key, None),
                        'agent_model':             data[agent_key]['model'],
                        'cfg_model':               data['config'].get(model_key, None),
                        'total_score':             data[agent_key]['total_score'],
                        'total_cooperations':      data[agent_key]['total_cooperations'],
                        'overall_cooperation_rate': data[agent_key]['overall_cooperation_rate']
                    })
                    agent_idx += 1

                # Insert episodes and rounds (dynamic — same as ipd2)
                for episode_data in data['episodes']:
                    episode_num = episode_data['episode']
                    agent_idx   = 0

                    while f'agent_{agent_idx}' in episode_data:
                        agent_key = f'agent_{agent_idx}'

                        cur.execute("""
                            INSERT INTO ipd3.episodes (
                                results_id, agent_idx, episode,
                                score, cooperations, cooperation_rate, reflection
                            ) VALUES (
                                %(results_id)s, %(agent_idx)s, %(episode)s,
                                %(score)s, %(cooperations)s,
                                %(cooperation_rate)s, %(reflection)s
                            ) RETURNING episode_id
                        """, {
                            'results_id':       results_id,
                            'agent_idx':        agent_idx,
                            'episode':          episode_num,
                            'score':            episode_data[agent_key]['episode_score'],
                            'cooperations':     episode_data[agent_key]['cooperations'],
                            'cooperation_rate': episode_data[agent_key]['cooperation_rate'],
                            'reflection':       episode_data[agent_key]['reflection']
                        })
                        episode_id = cur.fetchone()['episode_id']

                        for round_data in episode_data['rounds']:
                            action_key   = f'agent_{agent_idx}_action'
                            reasoning_key = f'agent_{agent_idx}_reasoning'
                            payoff_key   = f'agent_{agent_idx}_payoff'
                            ep_score_key = f'agent_{agent_idx}_episode_score'

                            cur.execute("""
                                INSERT INTO ipd3.rounds (
                                    episode_id, round, action, payoff,
                                    ep_cumulative_score, reasoning
                                ) VALUES (
                                    %(episode_id)s, %(round)s, %(action)s,
                                    %(payoff)s, %(ep_cumulative_score)s, %(reasoning)s
                                )
                            """, {
                                'episode_id':          episode_id,
                                'round':               round_data['round'],
                                'action':              round_data[action_key],
                                'payoff':              round_data[payoff_key],
                                'ep_cumulative_score': round_data[ep_score_key],
                                'reasoning':           round_data[reasoning_key]
                            })

                        agent_idx += 1

            self.conn.commit()
            logging.info(f"Loaded {filepath} -> results_id={results_id}, user={researcher}")
            return (results_id, researcher)

        except psycopg.errors.UniqueViolation as e:
            self.conn.rollback()
            err_msg = f"Duplicate file skipped: {filepath} - {e}"
            logging.warning(err_msg)
            print(err_msg)
            return None

        except Exception as e:
            self.conn.rollback()
            err_msg = f"Failed to load {filepath} - {e}"
            logging.error(err_msg)
            print(err_msg)
            raise

    def load_batch(self, source, pattern='*.json', user_name='unknown'):
        if isinstance(source, list):
            filepaths = source
        else:
            filepaths = glob.glob(os.path.join(source, pattern))

        if not filepaths:
            logging.warning("No files to process")
            return {'loaded': [], 'skipped': [], 'failed': []}

        logging.info(f"Processing {len(filepaths)} files")
        results = {'loaded': [], 'skipped': [], 'failed': []}

        for filepath in sorted(filepaths):
            try:
                result = self.load_json(filepath, user_name)
                if result is not None:
                    results['loaded'].append((filepath, result[0], result[1]))
                else:
                    results['skipped'].append(filepath)
            except Exception as e:
                err_msg = f"Failed to load {filepath} - {e}"
                print(err_msg)
                results['failed'].append((filepath, str(e)))

        logging.info(f"Batch complete: {len(results['loaded'])} loaded, "
                     f"{len(results['skipped'])} skipped, "
                     f"{len(results['failed'])} failed")
        return results

    def get_files(self, path, user_name='unknown'):
        if os.path.isfile(path):
            return self.load_json(path, user_name)
        elif os.path.isdir(path):
            return self.load_batch(path, user_name=user_name)
        elif '*' in path or '?' in path:
            dirpath = os.path.dirname(path) or '.'
            pattern = os.path.basename(path)
            return self.load_batch(dirpath, pattern, user_name)
        else:
            logging.error(f"Path not found: {path}")
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load 3-agent IPD game data into PostgreSQL')
    parser.add_argument('--import', dest='import_path', nargs='*',
                        help='File(s), directory, or pattern to load')
    parser.add_argument('--username', dest='user_name', default='unknown')

    args = parser.parse_args()

    if args.import_path:
        db = ForgeDB()

        if len(args.import_path) == 1:
            result = db.get_files(args.import_path[0], args.user_name)
            if isinstance(result, tuple):
                print(f"Loaded: results_id {result[0]}, user {result[1]}")
            elif isinstance(result, dict):
                print(f"Loaded: {len(result['loaded'])}, "
                      f"Skipped: {len(result['skipped'])}, "
                      f"Failed: {len(result['failed'])}")
        else:
            results = db.load_batch(args.import_path, user_name=args.user_name)
            print(f"Loaded: {len(results['loaded'])}, "
                  f"Skipped: {len(results['skipped'])}, "
                  f"Failed: {len(results['failed'])}")

        db.close()
    else:
        parser.print_help()
