/******************************************************************************
 * Practicum II - FORGE IPD3 Schema Setup
 * Persistent Storage Schema for 3-Agent Iterated Prisoner's Dilemma with LLM Agents
 *

 *
 * Changes from ipd2 schema (2-agent):
 *   - New schema name: ipd3 (to coexist with ipd2 on the same database)
 *   - Tables ipd3.results, ipd3.llm_agents, ipd3.episodes, ipd3.rounds,
 *     ipd3.research_log: structurally identical to ipd2 equivalents —
 *     the per-agent data is normalized into rows (agent_idx 0/1/2), so
 *     the tables themselves are agent-count agnostic.
 *   - SQL VIEWS updated:
 *       experiment_summary_vw: now JOINs agent_idx 0, 1, AND 2 explicitly
 *       episode_summary_vw:    now includes agent_2_* columns
 *       rounds_summary_vw:     now includes agent_2_* columns
 *       rounds_detail_vw:      unchanged (agent-agnostic by design)
 *       raw_data_vw:           unchanged
 *       results_vw:            unchanged
 ******************************************************************************/

CREATE SCHEMA ipd3;

/***************************** Create the Tables ******************************/

-- Stores one row per experiment run
CREATE TABLE ipd3.results (
  results_id                    SERIAL PRIMARY KEY
  ,filename                     VARCHAR(128) UNIQUE
  ,timestamp                    TIMESTAMPTZ UNIQUE
  ,hostname                     VARCHAR(64)
  ,username                     VARCHAR(64)
  ,elapsed_seconds              DOUBLE PRECISION
  ,cfg_num_episodes             SMALLINT
  ,cfg_round_per_episode        SMALLINT
  ,cfg_total_rounds             INTEGER
  ,cfg_history_window_size      SMALLINT
  ,cfg_temperature              REAL
  ,cfg_reset_between_episodes   BOOL
  ,cfg_reflection_type          VARCHAR(64)
  ,cfg_decision_token_limit     INTEGER
  ,cfg_reflection_token_limit   INTEGER
  ,cfg_http_timeout             SMALLINT
  ,cfg_force_decision_retries   SMALLINT
  ,system_prompt                TEXT
  ,reflection_template          TEXT
  ,raw_json                     JSONB
  ,comment                      TEXT
);

-- One row per agent per experiment (agent_idx = 0, 1, or 2)
CREATE TABLE ipd3.llm_agents (
  results_id                INTEGER
  ,agent_idx                SMALLINT
  ,host                     VARCHAR(64)
  ,agent_model              VARCHAR(64)
  ,cfg_model                VARCHAR(64)
  ,total_score              SMALLINT
  ,total_cooperations       SMALLINT
  ,overall_cooperation_rate REAL

  ,PRIMARY KEY (results_id, agent_idx)
  ,FOREIGN KEY (results_id)
        REFERENCES ipd3.results(results_id) ON DELETE CASCADE
);

-- One row per agent per episode
CREATE TABLE ipd3.episodes (
  episode_id                SERIAL PRIMARY KEY
  ,results_id               INTEGER
  ,agent_idx                SMALLINT
  ,episode                  SMALLINT
  ,score                    SMALLINT
  ,cooperations             SMALLINT
  ,cooperation_rate         DOUBLE PRECISION
  ,reflection               TEXT

  ,FOREIGN KEY (results_id)
        REFERENCES ipd3.results(results_id) ON DELETE CASCADE
  ,FOREIGN KEY (results_id, agent_idx)
        REFERENCES ipd3.llm_agents(results_id, agent_idx) ON DELETE CASCADE
  ,UNIQUE (results_id, agent_idx, episode)
);

-- One row per agent per round
CREATE TABLE ipd3.rounds (
  episode_id                INTEGER
  ,round                    SMALLINT
  ,action                   VARCHAR(64)
  ,payoff                   SMALLINT
  ,ep_cumulative_score      SMALLINT
  ,reasoning                TEXT

  ,PRIMARY KEY (episode_id, round)
  ,FOREIGN KEY (episode_id)
        REFERENCES ipd3.episodes(episode_id) ON DELETE CASCADE
);

-- Research log (identical structure to ipd2)
CREATE TABLE ipd3.research_log (
  log_id                    SERIAL PRIMARY KEY
  ,create_dttm              TIMESTAMPTZ
  ,log_dttm                 TIMESTAMPTZ
  ,username                 VARCHAR(64)
  ,subject                  VARCHAR(256)
  ,remarks                  TEXT
  ,tags                     TEXT[]
);

/******************************** Grant Access ********************************/
GRANT USAGE ON SCHEMA ipd3
  TO techkgirl, dhart, ksorauf, priyankasaha205, theandyman,rayudu;

GRANT ALL ON ALL TABLES IN SCHEMA ipd3
  TO techkgirl, dhart, ksorauf, priyankasaha205, theandyman,rayudu;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ipd3
  TO techkgirl, dhart, ksorauf, priyankasaha205, theandyman,rayudu;

/******************************* SQL Views ************************************/

-- Raw data view (agent-agnostic, unchanged from ipd2)
CREATE OR REPLACE VIEW ipd3.raw_data_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,r.raw_json
FROM ipd3.results r
ORDER BY r.username, r.timestamp;


-- Full results view, one row per agent per round (agent-agnostic)
CREATE OR REPLACE VIEW ipd3.results_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,r.hostname
    ,r.elapsed_seconds
    ,r.cfg_num_episodes
    ,r.cfg_round_per_episode
    ,r.cfg_total_rounds
    ,r.cfg_history_window_size
    ,r.cfg_temperature
    ,r.cfg_reset_between_episodes
    ,r.cfg_reflection_type
    ,r.cfg_decision_token_limit
    ,r.cfg_reflection_token_limit
    ,r.cfg_http_timeout
    ,r.cfg_force_decision_retries
    ,r.system_prompt
    ,r.reflection_template
    ,a.agent_idx
    ,a.host                         AS agent_host
    ,a.agent_model
    ,a.cfg_model
    ,a.total_score
    ,a.total_cooperations
    ,a.overall_cooperation_rate
    ,e.episode_id
    ,e.episode
    ,rd.round
    ,CONCAT('agent_', a.agent_idx)  AS agent
    ,rd.action
    ,rd.payoff
    ,rd.ep_cumulative_score
    ,rd.reasoning
    ,e.score                        AS episode_score
    ,e.cooperations                 AS ep_cooperations
    ,e.cooperation_rate             AS ep_coop_rate
    ,e.reflection                   AS ep_reflection
FROM
    ipd3.results r
    JOIN ipd3.llm_agents a ON a.results_id = r.results_id
    JOIN ipd3.episodes e   ON e.results_id = r.results_id AND e.agent_idx = a.agent_idx
    JOIN ipd3.rounds rd    ON rd.episode_id = e.episode_id
ORDER BY
    r.timestamp, a.agent_idx, e.episode, rd.round;


-- Experiment-level summary — pivoted to 3 agent columns
CREATE OR REPLACE VIEW ipd3.experiment_summary_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,r.hostname
    ,r.elapsed_seconds
    -- Agent 0
    ,a0.host                        AS agent_0_host
    ,a0.agent_model                 AS agent_0_model
    ,a0.total_score                 AS agent_0_total_score
    ,a0.total_cooperations          AS agent_0_total_cooperations
    ,a0.overall_cooperation_rate    AS agent_0_cooperation_rate
    -- Agent 1
    ,a1.host                        AS agent_1_host
    ,a1.agent_model                 AS agent_1_model
    ,a1.total_score                 AS agent_1_total_score
    ,a1.total_cooperations          AS agent_1_total_cooperations
    ,a1.overall_cooperation_rate    AS agent_1_cooperation_rate
    -- Agent 2 (NEW)
    ,a2.host                        AS agent_2_host
    ,a2.agent_model                 AS agent_2_model
    ,a2.total_score                 AS agent_2_total_score
    ,a2.total_cooperations          AS agent_2_total_cooperations
    ,a2.overall_cooperation_rate    AS agent_2_cooperation_rate
    ,r.cfg_num_episodes
    ,r.cfg_round_per_episode
    ,r.cfg_total_rounds
    ,r.cfg_history_window_size
    ,r.cfg_temperature
    ,r.cfg_reset_between_episodes
    ,r.cfg_reflection_type
    ,r.cfg_decision_token_limit
    ,r.cfg_reflection_token_limit
    ,r.cfg_http_timeout
    ,r.cfg_force_decision_retries
    ,r.system_prompt
    ,r.reflection_template
FROM
    ipd3.results r
    JOIN ipd3.llm_agents a0 ON a0.results_id = r.results_id AND a0.agent_idx = 0
    JOIN ipd3.llm_agents a1 ON a1.results_id = r.results_id AND a1.agent_idx = 1
    JOIN ipd3.llm_agents a2 ON a2.results_id = r.results_id AND a2.agent_idx = 2  -- NEW
ORDER BY r.timestamp;


-- Episode-level summary — pivoted to 3 agent columns
CREATE OR REPLACE VIEW ipd3.episode_summary_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,e0.episode
    -- Agent 0
    ,e0.score                   AS agent_0_episode_score
    ,e0.cooperations            AS agent_0_cooperations
    ,e0.cooperation_rate        AS agent_0_coop_rate
    ,e0.reflection              AS agent_0_reflection
    -- Agent 1
    ,e1.score                   AS agent_1_episode_score
    ,e1.cooperations            AS agent_1_cooperations
    ,e1.cooperation_rate        AS agent_1_coop_rate
    ,e1.reflection              AS agent_1_reflection
    -- Agent 2 (NEW)
    ,e2.score                   AS agent_2_episode_score
    ,e2.cooperations            AS agent_2_cooperations
    ,e2.cooperation_rate        AS agent_2_coop_rate
    ,e2.reflection              AS agent_2_reflection
FROM
    ipd3.results r
    JOIN ipd3.episodes e0 ON e0.results_id = r.results_id AND e0.agent_idx = 0
    JOIN ipd3.episodes e1 ON e1.results_id = r.results_id AND e1.agent_idx = 1 AND e1.episode = e0.episode
    JOIN ipd3.episodes e2 ON e2.results_id = r.results_id AND e2.agent_idx = 2 AND e2.episode = e0.episode  -- NEW
ORDER BY r.timestamp, e0.episode;


-- Round-level summary — pivoted to 3 agent columns
CREATE OR REPLACE VIEW ipd3.rounds_summary_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,a0.episode
    ,a0.round
    -- Agent 0
    ,a0.agent_0_episode_id
    ,a0.agent_0_action
    ,a0.agent_0_payoff
    ,a0.agent_0_ep_cumulative_score
    ,a0.agent_0_reasoning
    -- Agent 1
    ,a1.agent_1_episode_id
    ,a1.agent_1_action
    ,a1.agent_1_payoff
    ,a1.agent_1_ep_cumulative_score
    ,a1.agent_1_reasoning
    -- Agent 2 (NEW)
    ,a2.agent_2_episode_id
    ,a2.agent_2_action
    ,a2.agent_2_payoff
    ,a2.agent_2_ep_cumulative_score
    ,a2.agent_2_reasoning
FROM
    ipd3.results r
    JOIN (
        SELECT e.results_id, e.episode_id AS agent_0_episode_id, e.episode, rd.round,
               rd.action AS agent_0_action, rd.payoff AS agent_0_payoff,
               rd.ep_cumulative_score AS agent_0_ep_cumulative_score,
               rd.reasoning AS agent_0_reasoning
        FROM ipd3.episodes e JOIN ipd3.rounds rd ON rd.episode_id = e.episode_id
        WHERE e.agent_idx = 0
    ) a0 ON a0.results_id = r.results_id
    JOIN (
        SELECT e.results_id, e.episode_id AS agent_1_episode_id, e.episode, rd.round,
               rd.action AS agent_1_action, rd.payoff AS agent_1_payoff,
               rd.ep_cumulative_score AS agent_1_ep_cumulative_score,
               rd.reasoning AS agent_1_reasoning
        FROM ipd3.episodes e JOIN ipd3.rounds rd ON rd.episode_id = e.episode_id
        WHERE e.agent_idx = 1
    ) a1 ON a1.results_id = r.results_id AND a1.episode = a0.episode AND a1.round = a0.round
    JOIN (                                                                    -- NEW
        SELECT e.results_id, e.episode_id AS agent_2_episode_id, e.episode, rd.round,
               rd.action AS agent_2_action, rd.payoff AS agent_2_payoff,
               rd.ep_cumulative_score AS agent_2_ep_cumulative_score,
               rd.reasoning AS agent_2_reasoning
        FROM ipd3.episodes e JOIN ipd3.rounds rd ON rd.episode_id = e.episode_id
        WHERE e.agent_idx = 2
    ) a2 ON a2.results_id = r.results_id AND a2.episode = a0.episode AND a2.round = a0.round
ORDER BY r.timestamp, a0.episode, a0.round;


-- Detailed round view — agent-agnostic, unchanged from ipd2 pattern
CREATE OR REPLACE VIEW ipd3.rounds_detail_vw AS
SELECT
    r.results_id
    ,r.username
    ,r.filename
    ,r.comment
    ,r.timestamp
    ,e.episode_id
    ,a.agent_idx
    ,e.episode
    ,rd.round
    ,CONCAT('agent_', a.agent_idx) AS agent
    ,rd.action
    ,rd.payoff
    ,rd.ep_cumulative_score
    ,rd.reasoning
    ,e.score                        AS ep_score
    ,e.cooperations                 AS ep_cooperations
    ,e.cooperation_rate             AS ep_coop_rate
    ,e.reflection                   AS ep_reflection
FROM
    ipd3.results r
    JOIN ipd3.llm_agents a ON a.results_id = r.results_id
    JOIN ipd3.episodes e   ON e.results_id = r.results_id AND e.agent_idx = a.agent_idx
    JOIN ipd3.rounds rd    ON rd.episode_id = e.episode_id
ORDER BY r.timestamp, a.agent_idx, e.episode, rd.round;
