-- Reconciliation ledger with least-privilege roles.
-- Applying this file drops and recreates every schema it owns, so the data is disposable:
-- `recon load` regenerates it from the seeded synthetic businesses.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP SCHEMA IF EXISTS ledger, queue, audit, eval CASCADE;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'recon_agent') THEN
        CREATE ROLE recon_agent NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'recon_reviewer') THEN
        CREATE ROLE recon_reviewer NOLOGIN;
    END IF;
END
$$;

CREATE SCHEMA ledger;  -- the books: contacts, invoices, bank lines, posted allocations
CREATE SCHEMA queue;   -- what the rules pass and the agent propose
CREATE SCHEMA audit;   -- who approved what, append-only
CREATE SCHEMA eval;    -- ground-truth labels, never visible to the agent

CREATE TABLE ledger.contact (
    id    text PRIMARY KEY,
    name  text NOT NULL,
    payee text NOT NULL
);

CREATE TABLE ledger.invoice (
    id           text PRIMARY KEY,
    contact_id   text NOT NULL REFERENCES ledger.contact,
    issued       date NOT NULL,
    due          date NOT NULL,
    amount_cents bigint NOT NULL CHECK (amount_cents > 0),
    status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'paid'))
);
CREATE INDEX invoice_open_amount ON ledger.invoice (amount_cents) WHERE status = 'open';

CREATE TABLE ledger.bank_line (
    id           text PRIMARY KEY,
    posted       date NOT NULL,
    amount_cents bigint NOT NULL,
    payee        text NOT NULL,
    reference    text NOT NULL DEFAULT '',
    memo         text NOT NULL DEFAULT ''  -- written by the payer: untrusted
);

CREATE TABLE ledger.allocation (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bank_line_id text NOT NULL REFERENCES ledger.bank_line,
    invoice_id   text REFERENCES ledger.invoice,
    rule         text,
    posted_by    text NOT NULL DEFAULT current_user,
    posted_at    timestamptz NOT NULL DEFAULT now(),
    CHECK ((invoice_id IS NULL) <> (rule IS NULL))
);

CREATE TABLE queue.proposal (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id       text NOT NULL,
    bank_line_id text NOT NULL REFERENCES ledger.bank_line,
    decision     text NOT NULL CHECK (decision IN ('match', 'bank_rule', 'escalate')),
    invoice_ids  text[] NOT NULL DEFAULT '{}',
    rule         text NOT NULL DEFAULT '',
    source       text NOT NULL,  -- 'rules' or a model id
    rationale    text NOT NULL DEFAULT '',
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit.approval (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    proposal_id bigint NOT NULL REFERENCES queue.proposal,
    decision    text NOT NULL CHECK (decision IN ('approved', 'rejected', 'auto_approved')),
    actor       text NOT NULL DEFAULT current_user,
    note        text NOT NULL DEFAULT '',
    at          timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION audit.forbid_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit.% is append-only', TG_TABLE_NAME;
END
$$;
CREATE TRIGGER approval_no_update_or_delete BEFORE UPDATE OR DELETE ON audit.approval
    FOR EACH ROW EXECUTE FUNCTION audit.forbid_change();
CREATE TRIGGER approval_no_truncate BEFORE TRUNCATE ON audit.approval
    FOR EACH STATEMENT EXECUTE FUNCTION audit.forbid_change();

CREATE TABLE eval.label (
    line_id        text PRIMARY KEY REFERENCES ledger.bank_line,
    case_name      text NOT NULL,
    expected       text NOT NULL,
    invoice_ids    text[] NOT NULL,
    rule           text NOT NULL,
    attack_goal    text NOT NULL,
    attack_targets text[] NOT NULL,
    control        boolean NOT NULL,
    pii            jsonb NOT NULL
);

-- The agent reads the books and queues proposals. Nothing else.
GRANT USAGE ON SCHEMA ledger, queue TO recon_agent;
GRANT SELECT ON ledger.contact, ledger.invoice, ledger.bank_line TO recon_agent;
GRANT INSERT ON queue.proposal TO recon_agent;

-- The reviewer approves proposals, posts allocations and marks invoices paid.
GRANT USAGE ON SCHEMA ledger, queue, audit TO recon_reviewer;
GRANT SELECT ON ALL TABLES IN SCHEMA ledger, queue TO recon_reviewer;
GRANT INSERT ON ledger.allocation TO recon_reviewer;
GRANT UPDATE (status) ON ledger.invoice TO recon_reviewer;
GRANT SELECT, INSERT ON audit.approval TO recon_reviewer;
