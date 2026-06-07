--
-- PostgreSQL database dump
--

\restrict 7bR4RSJzoN5QL6B3sU6sbFlh7orIweMXFJwNdhmQnuA23a9SqX4s0gqb9hudugh

-- Dumped from database version 18.0
-- Dumped by pg_dump version 18.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: audit_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.audit_logs (
    id uuid NOT NULL,
    company_id uuid,
    actor_user_id uuid,
    actor_role character varying(50),
    action character varying(100) NOT NULL,
    target_type character varying(100),
    target_id character varying(255),
    detail jsonb,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.audit_logs OWNER TO postgres;

--
-- Name: checkpoints; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.checkpoints (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    session_id uuid NOT NULL,
    turn_start integer NOT NULL,
    turn_end integer NOT NULL,
    summary text NOT NULL,
    key_points jsonb DEFAULT '[]'::jsonb,
    customer_preferences jsonb DEFAULT '{}'::jsonb,
    objections_raised jsonb DEFAULT '[]'::jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.checkpoints OWNER TO postgres;

--
-- Name: companies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.companies (
    id uuid NOT NULL,
    name character varying(255) NOT NULL,
    slug character varying(120),
    is_active boolean,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.companies OWNER TO postgres;

--
-- Name: emotion_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.emotion_logs (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    session_id uuid NOT NULL,
    message_id uuid,
    customer_emotion character varying(50) NOT NULL,
    customer_mood_score integer,
    risk_level character varying(50),
    emotion_trend character varying(50),
    tip_shown text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.emotion_logs OWNER TO postgres;

--
-- Name: messages; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.messages (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    session_id uuid NOT NULL,
    turn_number integer NOT NULL,
    speaker character varying(50) NOT NULL,
    text text NOT NULL,
    audio_path character varying(500),
    audio_duration_seconds double precision,
    detected_emotion character varying(50),
    emotion_confidence double precision,
    emotion_scores jsonb,
    response_quality character varying(50),
    quality_reason text,
    suggestion text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms integer
);


ALTER TABLE public.messages OWNER TO postgres;

--
-- Name: personas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.personas (
    id character varying(100) NOT NULL,
    name_ar character varying(255) NOT NULL,
    name_en character varying(255) NOT NULL,
    description_ar text,
    description_en text,
    personality_prompt text NOT NULL,
    difficulty character varying(50) NOT NULL,
    patience_level integer DEFAULT 50,
    emotion_sensitivity integer DEFAULT 50,
    traits jsonb DEFAULT '[]'::jsonb,
    voice_id character varying(100),
    avatar_url character varying(500),
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.personas OWNER TO postgres;

--
-- Name: plans; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.plans (
    name character varying(50) NOT NULL,
    display_name character varying(100),
    seat_limit integer,
    session_limit_monthly integer,
    gaas_enabled boolean,
    price_monthly_usd integer,
    price_annual_usd integer
);


ALTER TABLE public.plans OWNER TO postgres;

--
-- Name: seat_invites; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.seat_invites (
    id uuid NOT NULL,
    company_id uuid NOT NULL,
    email character varying(255) NOT NULL,
    role character varying(50),
    token character varying(128) NOT NULL,
    status character varying(20),
    invited_by uuid,
    expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.seat_invites OWNER TO postgres;

--
-- Name: sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sessions (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    persona_id character varying(100) NOT NULL,
    status character varying(50) DEFAULT 'active'::character varying,
    difficulty character varying(50) NOT NULL,
    started_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    ended_at timestamp with time zone,
    duration_seconds integer,
    overall_score integer,
    communication_score integer,
    product_knowledge_score integer,
    objection_handling_score integer,
    rapport_score integer,
    closing_score integer,
    strengths jsonb DEFAULT '[]'::jsonb,
    weaknesses jsonb DEFAULT '[]'::jsonb,
    recommendations jsonb DEFAULT '[]'::jsonb,
    turn_count integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.sessions OWNER TO postgres;

--
-- Name: subscriptions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.subscriptions (
    id uuid NOT NULL,
    company_id uuid NOT NULL,
    plan_name character varying(50) NOT NULL,
    billing_cycle character varying(20),
    billing_status character varying(30),
    trial_ends_at timestamp with time zone,
    current_period_end timestamp with time zone,
    stripe_customer_id character varying(255),
    stripe_subscription_id character varying(255),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.subscriptions OWNER TO postgres;

--
-- Name: usage_periods; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.usage_periods (
    id uuid NOT NULL,
    company_id uuid NOT NULL,
    period_start date NOT NULL,
    sessions_used integer,
    seats_peak integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.usage_periods OWNER TO postgres;

--
-- Name: user_stats; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_stats (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    total_sessions integer DEFAULT 0,
    completed_sessions integer DEFAULT 0,
    total_training_minutes integer DEFAULT 0,
    avg_overall_score double precision,
    avg_communication_score double precision,
    avg_product_knowledge_score double precision,
    avg_objection_handling_score double precision,
    avg_rapport_score double precision,
    avg_closing_score double precision,
    best_session_id uuid,
    best_score integer,
    current_streak integer DEFAULT 0,
    longest_streak integer DEFAULT 0,
    last_session_date date,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.user_stats OWNER TO postgres;

--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    email character varying(255) NOT NULL,
    password_hash character varying(255) NOT NULL,
    full_name character varying(255) NOT NULL,
    company character varying(255),
    role character varying(50) DEFAULT 'salesperson'::character varying,
    experience_level character varying(50) DEFAULT 'beginner'::character varying,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    company_id uuid
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
0002_audit_logs
\.


--
-- Data for Name: audit_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.audit_logs (id, company_id, actor_user_id, actor_role, action, target_type, target_id, detail, created_at) FROM stdin;
c4386e1d-413d-4e57-8aff-966de702c5cb	66eb2663-63fc-406c-a7b7-ea577cd0ecb3	2715cc18-b273-4a01-9021-018b6d8f12d0	manager	company.created	company	66eb2663-63fc-406c-a7b7-ea577cd0ecb3	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:09:28.897349+03
f0509335-d815-4b28-abc8-814f90a8e46e	66eb2663-63fc-406c-a7b7-ea577cd0ecb3	2715cc18-b273-4a01-9021-018b6d8f12d0	manager	seat.invited	seat_invite	a1_3744410b@x.com	{"role": "salesperson", "email": "a1_3744410b@x.com"}	2026-06-07 19:09:29.322687+03
e2cfcb5f-7172-4808-825c-30349d0197c6	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	company.created	company	2aa5f809-d322-4d3d-85c6-1aeeac22539e	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:11:14.262462+03
2d75574c-929a-40b0-ac34-a94d5fd0440d	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.invited	seat_invite	a1_c19ff627@x.com	{"role": "salesperson", "email": "a1_c19ff627@x.com"}	2026-06-07 19:11:15.090248+03
5b6aad17-d933-4392-aab7-dbaf7dfd2857	2aa5f809-d322-4d3d-85c6-1aeeac22539e	4d010c16-1ec8-4eeb-9527-7ec1751590d0	salesperson	seat.accepted	user	4d010c16-1ec8-4eeb-9527-7ec1751590d0	{"email": "a1_c19ff627@x.com"}	2026-06-07 19:11:15.151654+03
2bbccf0c-ac44-48ec-a7e8-29e3ed6aaf3b	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.invited	seat_invite	a2_c19ff627@x.com	{"role": "salesperson", "email": "a2_c19ff627@x.com"}	2026-06-07 19:11:15.99985+03
248fbe41-6907-4523-bf86-7c98173dbc95	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.invited	seat_invite	a3_c19ff627@x.com	{"role": "salesperson", "email": "a3_c19ff627@x.com"}	2026-06-07 19:11:16.043321+03
afa56a0e-6a37-45f1-bbca-4fec8f691c86	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.invited	seat_invite	a4_c19ff627@x.com	{"role": "salesperson", "email": "a4_c19ff627@x.com"}	2026-06-07 19:11:16.0869+03
b348cce4-51c6-44e6-88b4-4db2b02c1221	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.invited	seat_invite	a5_c19ff627@x.com	{"role": "salesperson", "email": "a5_c19ff627@x.com"}	2026-06-07 19:11:16.141216+03
f2fc2741-5c61-4d3f-a050-8e707bdecc26	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	seat.deactivated	user	4d010c16-1ec8-4eeb-9527-7ec1751590d0	{"email": "a1_c19ff627@x.com"}	2026-06-07 19:11:16.238312+03
b90211bf-5edf-4394-b4b6-2d9735dea5b3	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	subscription.changed	subscription	4d0660bb-7209-4164-9f3b-0b10e3c88227	{"to": "free", "from": "starter", "billing_cycle": "monthly"}	2026-06-07 19:11:16.292094+03
44115fae-fdd1-4953-b09e-e15ec3f127d2	2aa5f809-d322-4d3d-85c6-1aeeac22539e	861aa3f9-5baf-4342-ac97-9b0ada0eda96	manager	subscription.changed	subscription	4d0660bb-7209-4164-9f3b-0b10e3c88227	{"to": "growth", "from": "free", "billing_cycle": "annual"}	2026-06-07 19:11:16.354578+03
74402164-0e4a-44d2-84a5-9cd7bbec6bdf	50c41bbb-26cc-46b8-9c06-20a8cd91b2f4	0de77f38-a75a-45e9-b270-45ea10eb00ad	manager	company.created	company	50c41bbb-26cc-46b8-9c06-20a8cd91b2f4	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:12:53.54186+03
67d81057-1f3f-41b3-b459-196b6c864614	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ad11f0b9-1b46-4569-8d0a-42c217f4d056	manager	company.created	company	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:13:43.993885+03
2fd1d0df-dc4f-40b4-92c7-ca2a317c0371	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ad11f0b9-1b46-4569-8d0a-42c217f4d056	manager	seat.invited	seat_invite	ag0_f7d91306@x.com	{"role": "salesperson", "email": "ag0_f7d91306@x.com"}	2026-06-07 19:13:44.86109+03
a321caa1-ad9b-470b-9ca5-e778a94bb9ed	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	365f96ec-df61-4b26-bb30-5486858b0dcf	salesperson	seat.accepted	user	365f96ec-df61-4b26-bb30-5486858b0dcf	{"email": "ag0_f7d91306@x.com"}	2026-06-07 19:13:44.948072+03
00daf9f8-762b-480d-b92a-768d7f426cd7	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ad11f0b9-1b46-4569-8d0a-42c217f4d056	manager	seat.invited	seat_invite	ag1_f7d91306@x.com	{"role": "salesperson", "email": "ag1_f7d91306@x.com"}	2026-06-07 19:13:45.82858+03
00ff0a7c-e636-4bbf-aa70-570b744be5d1	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	1cf3cfdb-d9ac-4d27-b56d-832e099b2781	salesperson	seat.accepted	user	1cf3cfdb-d9ac-4d27-b56d-832e099b2781	{"email": "ag1_f7d91306@x.com"}	2026-06-07 19:13:45.89101+03
22aedfb6-c863-4161-b818-f50e2943bceb	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ad11f0b9-1b46-4569-8d0a-42c217f4d056	manager	seat.invited	seat_invite	rev_f7d91306@x.com	{"role": "salesperson", "email": "rev_f7d91306@x.com"}	2026-06-07 19:13:46.712859+03
7ee20d54-c964-4b1d-9d3d-c10ac376b119	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ad11f0b9-1b46-4569-8d0a-42c217f4d056	manager	seat.revoked	seat_invite	bb3f0e33-f1d0-4926-8740-d3a42d3ad5b2	{"email": "rev_f7d91306@x.com"}	2026-06-07 19:13:46.759328+03
e68bd4bd-a617-483a-938c-99d326e00e7a	7920d481-911e-43a8-8778-db740e193687	c484ccea-ba5f-42ff-94d5-2b07274e86d3	manager	company.created	company	7920d481-911e-43a8-8778-db740e193687	{"plan_name": "scale", "billing_cycle": "monthly"}	2026-06-07 19:15:00.302097+03
d33e6527-c976-41b5-b6e4-9f71dded10ff	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	company.created	company	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:15:04.021433+03
63b6d8d4-b274-4f0b-8508-96a479283a0c	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	a1_59da9345@x.com	{"role": "salesperson", "email": "a1_59da9345@x.com"}	2026-06-07 19:15:04.937295+03
8af25f2e-f98e-49ef-bd42-344c8541e3a0	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	3723d638-6f27-4f92-9164-4ff388bf0eef	salesperson	seat.accepted	user	3723d638-6f27-4f92-9164-4ff388bf0eef	{"email": "a1_59da9345@x.com"}	2026-06-07 19:15:05.012425+03
6fc55658-8a40-4212-b517-79353683e43a	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	a2_59da9345@x.com	{"role": "salesperson", "email": "a2_59da9345@x.com"}	2026-06-07 19:15:05.759612+03
b4cfffdf-ecfd-4b1b-8811-6647a6b2af7c	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	a3_59da9345@x.com	{"role": "salesperson", "email": "a3_59da9345@x.com"}	2026-06-07 19:15:05.818672+03
f2df56d1-03a6-41f2-bd3b-ca996adc3133	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	a4_59da9345@x.com	{"role": "salesperson", "email": "a4_59da9345@x.com"}	2026-06-07 19:15:05.867966+03
7a04e357-777a-4cdd-aad9-244596f69096	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	a5_59da9345@x.com	{"role": "salesperson", "email": "a5_59da9345@x.com"}	2026-06-07 19:15:05.916426+03
cf59afeb-3261-4140-addf-58e4ce5f4b46	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.deactivated	user	3723d638-6f27-4f92-9164-4ff388bf0eef	{"email": "a1_59da9345@x.com"}	2026-06-07 19:15:06.006633+03
5fdbeec2-5cdf-4c73-a9de-3432757b3c10	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	ec3fb8d3-1217-40b1-9233-263937c19f9b	manager	seat.invited	seat_invite	b0_59da9345@x.com	{"role": "salesperson", "email": "b0_59da9345@x.com"}	2026-06-07 19:15:06.043659+03
941df722-df69-449d-bf78-eda685ab51d2	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	b5baa5d9-7b71-4792-9cbd-3a360368212b	salesperson	seat.accepted	user	b5baa5d9-7b71-4792-9cbd-3a360368212b	{"email": "b0_59da9345@x.com"}	2026-06-07 19:15:06.082712+03
d65d9e37-0dad-439a-9c31-9fac37e92d3e	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	company.created	company	25709324-afa8-415b-a277-3a00eaf144aa	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:16:06.272556+03
5a61b08b-cf8e-4670-8bbb-05dd33e3e36d	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.invited	seat_invite	a1_91266b9f@x.com	{"role": "salesperson", "email": "a1_91266b9f@x.com"}	2026-06-07 19:16:07.24107+03
0175b050-e8d5-4a3e-be21-4ba84832a645	25709324-afa8-415b-a277-3a00eaf144aa	2c878a76-c715-41b2-87fa-6fc7cc862f88	salesperson	seat.accepted	user	2c878a76-c715-41b2-87fa-6fc7cc862f88	{"email": "a1_91266b9f@x.com"}	2026-06-07 19:16:07.30917+03
43753b95-18be-4330-9926-eded7b67c6f1	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.invited	seat_invite	a2_91266b9f@x.com	{"role": "salesperson", "email": "a2_91266b9f@x.com"}	2026-06-07 19:16:08.032443+03
8830b122-644c-4902-9287-39445f7a9ba0	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.invited	seat_invite	a3_91266b9f@x.com	{"role": "salesperson", "email": "a3_91266b9f@x.com"}	2026-06-07 19:16:08.068824+03
42e4287d-e299-44d7-b9c7-dd47905cb044	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.invited	seat_invite	a4_91266b9f@x.com	{"role": "salesperson", "email": "a4_91266b9f@x.com"}	2026-06-07 19:16:08.108851+03
d9672c6a-f7ed-4e63-a4dd-fed3f66f34e4	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.invited	seat_invite	a5_91266b9f@x.com	{"role": "salesperson", "email": "a5_91266b9f@x.com"}	2026-06-07 19:16:08.158229+03
55324c46-d5d7-44ef-b382-6b09814add80	25709324-afa8-415b-a277-3a00eaf144aa	348abbd9-92ab-4e7c-89ed-3b69de4df74a	manager	seat.deactivated	user	2c878a76-c715-41b2-87fa-6fc7cc862f88	{"email": "a1_91266b9f@x.com"}	2026-06-07 19:16:08.249253+03
8ecffb09-65f9-4567-80fd-4e2d1fc314e7	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	399b588b-462b-4352-a691-30ae3fc3a6ad	manager	company.created	company	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	{"plan_name": "starter", "billing_cycle": "monthly"}	2026-06-07 19:16:08.287317+03
bb009a23-901f-4a0b-9139-b490234252d3	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	399b588b-462b-4352-a691-30ae3fc3a6ad	manager	seat.invited	seat_invite	b0_da9e7b0b@x.com	{"role": "salesperson", "email": "b0_da9e7b0b@x.com"}	2026-06-07 19:16:09.006599+03
bd861b99-df7e-4bd0-a5a3-2c1c005c291d	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	c8046705-322b-4c07-a682-c6b3a5c14f89	salesperson	seat.accepted	user	c8046705-322b-4c07-a682-c6b3a5c14f89	{"email": "b0_da9e7b0b@x.com"}	2026-06-07 19:16:09.050211+03
184ef0e8-9099-4b28-abcd-e985a0809143	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	399b588b-462b-4352-a691-30ae3fc3a6ad	manager	seat.invited	seat_invite	b1_da9e7b0b@x.com	{"role": "salesperson", "email": "b1_da9e7b0b@x.com"}	2026-06-07 19:16:09.748646+03
744aa3cd-f24c-4ad4-88b2-71424f6ff762	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	0140a01c-2d01-4bf7-b146-8556b6309b34	salesperson	seat.accepted	user	0140a01c-2d01-4bf7-b146-8556b6309b34	{"email": "b1_da9e7b0b@x.com"}	2026-06-07 19:16:09.788341+03
49ed6bee-0683-417d-9c77-49a35d46cf64	10842b3e-ef84-4bca-9282-eb40d29be97d	4e45d795-ae16-4cf2-b9c6-ba173335711a	manager	company.created	company	10842b3e-ef84-4bca-9282-eb40d29be97d	{"plan_name": "free", "billing_cycle": "monthly"}	2026-06-07 19:50:57.027234+03
\.


--
-- Data for Name: checkpoints; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.checkpoints (id, session_id, turn_start, turn_end, summary, key_points, customer_preferences, objections_raised, created_at) FROM stdin;
\.


--
-- Data for Name: companies; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.companies (id, name, slug, is_active, created_at, updated_at) FROM stdin;
324ef612-b8a3-402c-865b-2e3317873dad	A	a	t	2026-06-07 18:32:33.516748+03	2026-06-07 18:32:33.516748+03
2331807a-ab61-4c5c-948a-00a1db3de6e9	VCAI Test Company	vcai-test	t	2026-06-07 18:35:06.99221+03	2026-06-07 18:35:06.99221+03
66eb2663-63fc-406c-a7b7-ea577cd0ecb3	Acme 3744410b	acme-3744410b	t	2026-06-07 19:09:28.897349+03	2026-06-07 19:09:28.897349+03
2aa5f809-d322-4d3d-85c6-1aeeac22539e	Acme c19ff627	acme-c19ff627	t	2026-06-07 19:11:14.262462+03	2026-06-07 19:11:14.262462+03
50c41bbb-26cc-46b8-9c06-20a8cd91b2f4	Beta 592cd5df	beta-592cd5df	t	2026-06-07 19:12:53.54186+03	2026-06-07 19:12:53.54186+03
1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	Beta f7d91306	beta-f7d91306	t	2026-06-07 19:13:43.993885+03	2026-06-07 19:13:43.993885+03
7920d481-911e-43a8-8778-db740e193687	Hegazi INC.	hegazi-inc	t	2026-06-07 19:15:00.302097+03	2026-06-07 19:15:00.302097+03
caaad1c6-ffa3-43e1-8ed7-949d08c486a7	Acme 59da9345	acme-59da9345	t	2026-06-07 19:15:04.021433+03	2026-06-07 19:15:04.021433+03
25709324-afa8-415b-a277-3a00eaf144aa	Acme 91266b9f	acme-91266b9f	t	2026-06-07 19:16:06.272556+03	2026-06-07 19:16:06.272556+03
5276805f-82f4-4c23-9a87-6b1b30c4f3f8	Beta da9e7b0b	beta-da9e7b0b	t	2026-06-07 19:16:08.287317+03	2026-06-07 19:16:08.287317+03
10842b3e-ef84-4bca-9282-eb40d29be97d	MIU	miu	t	2026-06-07 19:50:57.027234+03	2026-06-07 19:50:57.027234+03
\.


--
-- Data for Name: emotion_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.emotion_logs (id, session_id, message_id, customer_emotion, customer_mood_score, risk_level, emotion_trend, tip_shown, created_at) FROM stdin;
\.


--
-- Data for Name: messages; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.messages (id, session_id, turn_number, speaker, text, audio_path, audio_duration_seconds, detected_emotion, emotion_confidence, emotion_scores, response_quality, quality_reason, suggestion, created_at, processing_time_ms) FROM stdin;
\.


--
-- Data for Name: personas; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.personas (id, name_ar, name_en, description_ar, description_en, personality_prompt, difficulty, patience_level, emotion_sensitivity, traits, voice_id, avatar_url, is_active, created_at) FROM stdin;
difficult_customer	عميل صعب	Difficult Customer	عميل متشكك وبيفاصل كتير وصعب ترضيه	Skeptical customer who negotiates hard and is difficult to please	أنت عميل مصري صعب بتدور على شقة. أنت:\n- متشكك جداً في كل حاجة البائع بيقولها\n- بتفاصل بقوة على السعر\n- بتسأل أسئلة كتير ومحرجة\n- مش بتقتنع بسهولة\n- بتقارن بعروض تانية (حقيقية أو وهمية)\n- لو البائع كان وقح أو مش محترف، بتزعل جداً وممكن تمشي	hard	30	80	["متشكك", "بيفاصل", "صعب الإرضاء", "كتير الأسئلة"]	egyptian_male_01	\N	t	2026-02-06 21:09:54.009192+02
friendly_customer	عميل ودود	Friendly Customer	عميل لطيف ومتعاون وسهل التعامل معاه	Friendly and cooperative customer, easy to work with	أنت عميل مصري ودود بتدور على شقة. أنت:\n- لطيف ومتعاون مع البائع\n- بتسمع كويس وبتسأل أسئلة منطقية\n- منفتح على الاقتراحات\n- بتقدر المعلومات المفيدة\n- لو البائع كان كويس، بتكون إيجابي معاه\n- حتى لو في مشكلة، بتتعامل بهدوء	easy	80	30	["ودود", "متعاون", "صبور", "منفتح"]	egyptian_male_02	\N	t	2026-02-06 21:09:54.009192+02
rushed_customer	عميل مستعجل	Rushed Customer	عميل مشغول ووقته ضيق وعايز يخلص بسرعة	Busy customer with limited time who wants quick answers	أنت عميل مصري مستعجل بتدور على شقة. أنت:\n- وقتك ضيق جداً ومشغول\n- عايز إجابات مباشرة وسريعة\n- بتزهق من الكلام الكتير\n- بتقاطع لو البائع طول في الشرح\n- عايز تعرف السعر والموقع والمساحة بس\n- لو البائع ضيع وقتك، بتمشي	medium	40	60	["مستعجل", "مباشر", "عملي", "قليل الصبر"]	egyptian_male_01	\N	t	2026-02-06 21:09:54.009192+02
price_focused_customer	عميل مهتم بالسعر	Price-Focused Customer	عميل ميزانيته محدودة وبيدور على أحسن سعر	Budget-conscious customer looking for the best deal	أنت عميل مصري مهتم بالسعر بتدور على شقة. أنت:\n- ميزانيتك محدودة جداً\n- كل حاجة بتقيسها بالفلوس\n- بتسأل عن التقسيط والتسهيلات\n- بتقارن الأسعار بالسوق\n- بتفاصل على كل حاجة\n- مستعد تتنازل عن حاجات مقابل سعر أقل	medium	50	50	["حريص", "بيفاصل", "عملي", "مقارن"]	egyptian_female_01	\N	t	2026-02-06 21:09:54.009192+02
first_time_buyer	مشتري لأول مرة	First-Time Buyer	عميل بيشتري لأول مرة ومحتاج توجيه ومساعدة	First-time buyer who needs guidance and help	أنت عميل مصري بتشتري شقة لأول مرة. أنت:\n- مش فاهم كتير في العقارات\n- محتاج حد يشرحلك كل حاجة\n- بتسأل أسئلة بسيطة (ممكن تبان ساذجة)\n- قلقان من الغش أو الخداع\n- محتاج طمأنة ومعلومات واضحة\n- بتقدر البائع اللي بيساعدك فعلاً	easy	70	40	["محتاج توجيه", "قلقان", "بيسأل كتير", "بيثق بسهولة"]	egyptian_female_01	\N	t	2026-02-06 21:09:54.009192+02
\.


--
-- Data for Name: plans; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.plans (name, display_name, seat_limit, session_limit_monthly, gaas_enabled, price_monthly_usd, price_annual_usd) FROM stdin;
free	Free	1	5	f	0	0
starter	Starter	5	30	f	29	290
growth	Growth	20	150	t	99	990
scale	Scale	100	1000000	t	299	2990
enterprise	Enterprise	1000000	1000000	t	\N	\N
\.


--
-- Data for Name: seat_invites; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.seat_invites (id, company_id, email, role, token, status, invited_by, expires_at, created_at) FROM stdin;
c0c43374-1088-486d-be6a-23de23f5dce0	66eb2663-63fc-406c-a7b7-ea577cd0ecb3	a1_3744410b@x.com	salesperson	EpI6vvovnU8IfTpnNKl7-ArctKZ8gzXV8EG36Q9iI5g	pending	2715cc18-b273-4a01-9021-018b6d8f12d0	2026-06-14 16:09:29.337349+03	2026-06-07 19:09:29.322687+03
86dbcfa5-86a7-46c6-8654-738c72512874	2aa5f809-d322-4d3d-85c6-1aeeac22539e	a1_c19ff627@x.com	salesperson	N1NkozBcBd6asTy-ElAHjhBrN7EKS4-r3Mmabd1tgXo	accepted	861aa3f9-5baf-4342-ac97-9b0ada0eda96	2026-06-14 19:11:15.113839+03	2026-06-07 19:11:15.090248+03
2bff8ead-ea94-4c5f-81dd-d0726c828131	2aa5f809-d322-4d3d-85c6-1aeeac22539e	a2_c19ff627@x.com	salesperson	lpI46NifanwRH1PG3Yvvk3aT92MwJ_X-vc1wOpW84P4	pending	861aa3f9-5baf-4342-ac97-9b0ada0eda96	2026-06-14 19:11:16.013293+03	2026-06-07 19:11:15.99985+03
8baa9542-d8d7-4c1a-accc-e79e0e98f6d5	2aa5f809-d322-4d3d-85c6-1aeeac22539e	a3_c19ff627@x.com	salesperson	PVN8J_t7iKijaMYzrQui3vclalb1tPJIHb_cg8zf-Ok	pending	861aa3f9-5baf-4342-ac97-9b0ada0eda96	2026-06-14 19:11:16.056206+03	2026-06-07 19:11:16.043321+03
9a0a6fd0-0264-4f37-8c19-7e5d818b92d8	2aa5f809-d322-4d3d-85c6-1aeeac22539e	a4_c19ff627@x.com	salesperson	ZEQY_RkpaL8wlK2BgVddoJKB5DdKXtoDZITWo5v5yBA	pending	861aa3f9-5baf-4342-ac97-9b0ada0eda96	2026-06-14 19:11:16.101685+03	2026-06-07 19:11:16.0869+03
6b27101f-97f7-411d-93ac-d6f0378d1b16	2aa5f809-d322-4d3d-85c6-1aeeac22539e	a5_c19ff627@x.com	salesperson	0ZABI7rUVLROcvq4F7nB3LzF_bgjAXy2gRtEKgG5G24	pending	861aa3f9-5baf-4342-ac97-9b0ada0eda96	2026-06-14 19:11:16.158251+03	2026-06-07 19:11:16.141216+03
f8a8bb30-c080-469c-80ad-65bc095de02c	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ag0_f7d91306@x.com	salesperson	_YLWcS1dmsomY1lVsF-RCzretc0O-Z3jmzUKeIkwLa4	accepted	ad11f0b9-1b46-4569-8d0a-42c217f4d056	2026-06-14 19:13:44.902594+03	2026-06-07 19:13:44.86109+03
84155aa4-fd12-46c0-8e4d-6f41feabd00c	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	ag1_f7d91306@x.com	salesperson	OlrD25jPGoPtIsJ8TXfZ8C15ixR_QW2ywS6-V0SGNUo	accepted	ad11f0b9-1b46-4569-8d0a-42c217f4d056	2026-06-14 19:13:45.848649+03	2026-06-07 19:13:45.82858+03
bb3f0e33-f1d0-4926-8740-d3a42d3ad5b2	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	rev_f7d91306@x.com	salesperson	FC1z9_a8vsRfbUUJ9KzWt3lBjdthS4vc0E_E5glOX7c	revoked	ad11f0b9-1b46-4569-8d0a-42c217f4d056	2026-06-14 19:13:46.72765+03	2026-06-07 19:13:46.712859+03
d49f98a8-816c-47cf-8f4f-c3b4013002d1	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	a1_59da9345@x.com	salesperson	-6j8f0DLZGs9q2BmGlmSihla7ZJ7zhZJPLwQ0fR_Tuc	accepted	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:04.966924+03	2026-06-07 19:15:04.937295+03
4b65fa36-e839-4e7e-9b79-bfdace9eaa71	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	a2_59da9345@x.com	salesperson	mM9VHYviBglz9rZxMLoTydkAvAZaehK8bB8W-OW1shc	pending	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:05.780017+03	2026-06-07 19:15:05.759612+03
816d50f8-441f-43e1-bc05-9fe50d786427	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	a3_59da9345@x.com	salesperson	opkdz0pg-pftTYeVJSAxxRvUx3pWM20CbQ9zBe-UJ3I	pending	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:05.834815+03	2026-06-07 19:15:05.818672+03
216ce5f7-d7db-4ebc-b6fd-ee2bc2460bce	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	a4_59da9345@x.com	salesperson	9gCffxAOwhVzxCmSFDZbngdcjqfh7zcgjWW36QdNHJg	pending	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:05.88108+03	2026-06-07 19:15:05.867966+03
e3c7ee0d-ebb4-4863-9e14-83a956966cf4	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	a5_59da9345@x.com	salesperson	tNsFRo3VkibI9Yf-et9StQei-vALs8LUH7mmvDIoR0Y	pending	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:05.932204+03	2026-06-07 19:15:05.916426+03
672fb985-15e3-4bb5-b248-deeb7ac8a456	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	b0_59da9345@x.com	salesperson	NlWJITe6ZGsBKjFvnBgZqOOP72J7EVbobwrnKEju_KM	accepted	ec3fb8d3-1217-40b1-9233-263937c19f9b	2026-06-14 19:15:06.055872+03	2026-06-07 19:15:06.043659+03
09bd74b5-2c69-4fd8-8699-29c557567f1e	25709324-afa8-415b-a277-3a00eaf144aa	a1_91266b9f@x.com	salesperson	tNgE-oh_2Ip5pgRX0wFh-yCHH9ML8LHbgK9Euiiqej4	accepted	348abbd9-92ab-4e7c-89ed-3b69de4df74a	2026-06-14 19:16:07.269361+03	2026-06-07 19:16:07.24107+03
12db76c7-eb76-4b4a-ae31-df1701594f49	25709324-afa8-415b-a277-3a00eaf144aa	a2_91266b9f@x.com	salesperson	VQCGF36XHwC0zNkuUNKzzkUc24VxV-HGdo16gyk_WmE	pending	348abbd9-92ab-4e7c-89ed-3b69de4df74a	2026-06-14 19:16:08.043814+03	2026-06-07 19:16:08.032443+03
aa917828-d73e-4512-b0b8-5d727468a0d4	25709324-afa8-415b-a277-3a00eaf144aa	a3_91266b9f@x.com	salesperson	H2MunhfYm3UskystMX04yeU3JTvJLPdjTBfWEO2iF9s	pending	348abbd9-92ab-4e7c-89ed-3b69de4df74a	2026-06-14 19:16:08.081295+03	2026-06-07 19:16:08.068824+03
a12e6ff4-4fd9-41b1-9690-f47cdfb9088d	25709324-afa8-415b-a277-3a00eaf144aa	a4_91266b9f@x.com	salesperson	MkRaVwLHNk7FSeXODTKUPrJXCOAQAd-AUsbkaEuAQaA	pending	348abbd9-92ab-4e7c-89ed-3b69de4df74a	2026-06-14 19:16:08.119593+03	2026-06-07 19:16:08.108851+03
5d5b486d-be91-4ac8-af88-0e3780fe06c6	25709324-afa8-415b-a277-3a00eaf144aa	a5_91266b9f@x.com	salesperson	BVKoIRpsbP-X0wkOkSnLWc_5TQyHvjMi87tLZ9oujiI	pending	348abbd9-92ab-4e7c-89ed-3b69de4df74a	2026-06-14 19:16:08.179375+03	2026-06-07 19:16:08.158229+03
aacd7b30-c50d-4eda-bcc0-32b9db3fb156	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	b0_da9e7b0b@x.com	salesperson	w-WPkzAYwN4FkdDTiQrt7s4jvRPZkVsrZl-BGgqvEsQ	accepted	399b588b-462b-4352-a691-30ae3fc3a6ad	2026-06-14 19:16:09.01997+03	2026-06-07 19:16:09.006599+03
fcec0cbb-b231-41d5-b265-9c0d275bb237	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	b1_da9e7b0b@x.com	salesperson	WEr6k4hB_3VeNuCcPbYhe_jKCbLKkSazMdQO2Th3pso	accepted	399b588b-462b-4352-a691-30ae3fc3a6ad	2026-06-14 19:16:09.761422+03	2026-06-07 19:16:09.748646+03
\.


--
-- Data for Name: sessions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sessions (id, user_id, persona_id, status, difficulty, started_at, ended_at, duration_seconds, overall_score, communication_score, product_knowledge_score, objection_handling_score, rapport_score, closing_score, strengths, weaknesses, recommendations, turn_count, created_at) FROM stdin;
3a255d2c-796c-4a5b-b7b2-9144d10c91bc	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	rushed_customer	completed	medium	2026-02-06 22:26:30.749089+02	2026-02-06 22:27:40.238387+02	69	\N	\N	\N	\N	\N	\N	[]	[]	[]	0	2026-02-06 22:26:30.749089+02
7222ce99-6d47-447a-99a3-d430b6c03b14	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	first_time_buyer	active	easy	2026-02-06 22:28:00.001121+02	\N	\N	\N	\N	\N	\N	\N	\N	[]	[]	[]	0	2026-02-06 22:28:00.001121+02
\.


--
-- Data for Name: subscriptions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.subscriptions (id, company_id, plan_name, billing_cycle, billing_status, trial_ends_at, current_period_end, stripe_customer_id, stripe_subscription_id, created_at, updated_at) FROM stdin;
81c0b85a-0822-41d5-b160-364b61491417	324ef612-b8a3-402c-865b-2e3317873dad	free	monthly	active	\N	\N	\N	\N	2026-06-07 18:32:33.516748+03	2026-06-07 18:32:33.516748+03
726234c8-91d4-4359-8431-9655088bb99e	2331807a-ab61-4c5c-948a-00a1db3de6e9	scale	annual	active	\N	\N	\N	\N	2026-06-07 18:35:06.99221+03	2026-06-07 18:35:06.99221+03
404a0c79-8db6-440c-881e-3bebf74b00e6	66eb2663-63fc-406c-a7b7-ea577cd0ecb3	starter	monthly	trial	2026-06-21 16:09:28.93401+03	\N	\N	\N	2026-06-07 19:09:28.897349+03	2026-06-07 19:09:28.897349+03
4d0660bb-7209-4164-9f3b-0b10e3c88227	2aa5f809-d322-4d3d-85c6-1aeeac22539e	growth	annual	trial	2026-06-21 19:11:14.281726+03	\N	\N	\N	2026-06-07 19:11:14.262462+03	2026-06-07 19:11:16.354578+03
974634d9-aa5b-4e03-9906-bf7e7bbd3710	50c41bbb-26cc-46b8-9c06-20a8cd91b2f4	starter	monthly	trial	2026-06-21 19:12:53.567091+03	\N	\N	\N	2026-06-07 19:12:53.54186+03	2026-06-07 19:12:53.54186+03
cee72b41-1197-4b5b-9057-c940b1c022fd	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3	starter	monthly	trial	2026-06-21 19:13:44.014315+03	\N	\N	\N	2026-06-07 19:13:43.993885+03	2026-06-07 19:13:43.993885+03
b7bf5bc4-d44d-483f-933b-0c88186a38ac	7920d481-911e-43a8-8778-db740e193687	scale	monthly	trial	2026-06-21 16:15:00.4151+03	\N	\N	\N	2026-06-07 19:15:00.302097+03	2026-06-07 19:15:00.302097+03
ff9e2a63-dfb3-48ce-8e2d-3d62c253a439	caaad1c6-ffa3-43e1-8ed7-949d08c486a7	starter	monthly	trial	2026-06-21 19:15:04.040978+03	\N	\N	\N	2026-06-07 19:15:04.021433+03	2026-06-07 19:15:04.021433+03
b76d14ef-a903-4679-823f-b1d4f933b8d7	25709324-afa8-415b-a277-3a00eaf144aa	starter	monthly	trial	2026-06-21 19:16:06.303668+03	\N	\N	\N	2026-06-07 19:16:06.272556+03	2026-06-07 19:16:06.272556+03
3327e3d1-07d4-4bfb-9c28-ec7610c96328	5276805f-82f4-4c23-9a87-6b1b30c4f3f8	starter	monthly	trial	2026-06-21 19:16:08.293606+03	\N	\N	\N	2026-06-07 19:16:08.287317+03	2026-06-07 19:16:08.287317+03
bc2b55f7-612d-4e81-9478-0964502afd48	10842b3e-ef84-4bca-9282-eb40d29be97d	free	monthly	active	\N	\N	\N	\N	2026-06-07 19:50:57.027234+03	2026-06-07 19:50:57.027234+03
\.


--
-- Data for Name: usage_periods; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.usage_periods (id, company_id, period_start, sessions_used, seats_peak, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: user_stats; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.user_stats (id, user_id, total_sessions, completed_sessions, total_training_minutes, avg_overall_score, avg_communication_score, avg_product_knowledge_score, avg_objection_handling_score, avg_rapport_score, avg_closing_score, best_session_id, best_score, current_streak, longest_streak, last_session_date, updated_at) FROM stdin;
f69c855a-c176-404c-80d0-26c252a151cc	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-02-06 22:26:18.343672+02
07743dfb-5f96-400f-b49e-e7097c4a7456	cbb2d078-65f9-4e57-887f-b9c3126924af	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:08:53.765699+03
c656fbb5-06f6-4da6-981e-fee5d1ed819b	2715cc18-b273-4a01-9021-018b6d8f12d0	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:09:28.897349+03
ef08f3b0-35c3-4f3f-b582-801ad0581bab	861aa3f9-5baf-4342-ac97-9b0ada0eda96	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:11:14.262462+03
2e0fec0c-e98d-4cf2-ac36-06dab03a6698	4d010c16-1ec8-4eeb-9527-7ec1751590d0	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:11:15.151654+03
f8575a39-2a93-466a-82a4-8d22963901ae	0de77f38-a75a-45e9-b270-45ea10eb00ad	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:12:53.54186+03
c6937047-838f-4cf3-ab3f-707f7f18f4c0	ad11f0b9-1b46-4569-8d0a-42c217f4d056	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:13:43.993885+03
6dc4c229-ccff-4391-8831-752942f7fb5d	365f96ec-df61-4b26-bb30-5486858b0dcf	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:13:44.948072+03
feac1fb7-6dd3-4a82-a2ab-114f2c623818	1cf3cfdb-d9ac-4d27-b56d-832e099b2781	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:13:45.89101+03
e51dc44b-397e-4af5-9d18-39c72806bf67	c484ccea-ba5f-42ff-94d5-2b07274e86d3	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:15:00.302097+03
fcc6f761-8257-4de6-b747-880de35af06c	ec3fb8d3-1217-40b1-9233-263937c19f9b	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:15:04.021433+03
b325e178-b6d5-40bc-9854-50227a53dc9d	3723d638-6f27-4f92-9164-4ff388bf0eef	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:15:05.012425+03
26a85c36-f8b2-4e09-b0c3-afc9cbb3132a	b5baa5d9-7b71-4792-9cbd-3a360368212b	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:15:06.082712+03
6f421e18-5d7a-428b-bf5f-0bd1424a8351	348abbd9-92ab-4e7c-89ed-3b69de4df74a	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:16:06.272556+03
d00ca497-14a7-487a-9d77-c04dfbafafd0	2c878a76-c715-41b2-87fa-6fc7cc862f88	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:16:07.30917+03
cd57c2e7-a9ad-49cf-83c6-0b3cd96c7200	399b588b-462b-4352-a691-30ae3fc3a6ad	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:16:08.287317+03
1e207bae-23a8-4d6e-8b1b-fa60c3242604	c8046705-322b-4c07-a682-c6b3a5c14f89	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:16:09.050211+03
2e0827e1-cc8b-4b56-a309-8d287b3aa3ec	0140a01c-2d01-4bf7-b146-8556b6309b34	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:16:09.788341+03
baaa67c7-fc15-45cc-b24f-b7ebf417f498	fc567435-4933-4b6f-842c-0c8da1d11a8a	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:31:34.870202+03
cb5eace3-e030-4d8a-87b2-34a0a2892c61	4e45d795-ae16-4cf2-b9c6-ba173335711a	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-06-07 19:50:57.027234+03
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, email, password_hash, full_name, company, role, experience_level, is_active, created_at, updated_at, company_id) FROM stdin;
3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	a@a.com	$2b$12$zzWsT21c/yQmMFbM6DJKuetEuf8APngWa3ZA05eFePqUWFg8DlkSC	Abubakr	A	salesperson	beginner	t	2026-02-06 22:26:18.06929+02	2026-06-07 18:32:33.516748+03	324ef612-b8a3-402c-865b-2e3317873dad
cbb2d078-65f9-4e57-887f-b9c3126924af	corstest@example.com	$2b$12$EA77GhxLfPA.H4pps6lGLOA/DuN.uVczBUxBhseZshgu4gZ9KOzhO	CORS Test	\N	salesperson	beginner	t	2026-06-07 19:08:53.309248+03	2026-06-07 19:08:53.309248+03	\N
2715cc18-b273-4a01-9021-018b6d8f12d0	mgr_3744410b@example.com	$2b$12$oDP290mkt073ElMgQmc/keYHBkLHIzHYaTWd7Eu9XfnDTLD3zB4x2	Jane Manager	Acme 3744410b	manager	beginner	t	2026-06-07 19:09:28.897349+03	2026-06-07 19:09:28.897349+03	66eb2663-63fc-406c-a7b7-ea577cd0ecb3
861aa3f9-5baf-4342-ac97-9b0ada0eda96	mgr_c19ff627@example.com	$2b$12$9ApRZRTmiaBwPp7yGi5Xb.kaeERMNKOzGa3v.ApdoRAAt73AOQy9y	Jane Manager	Acme c19ff627	manager	beginner	t	2026-06-07 19:11:14.262462+03	2026-06-07 19:11:14.262462+03	2aa5f809-d322-4d3d-85c6-1aeeac22539e
4d010c16-1ec8-4eeb-9527-7ec1751590d0	a1_c19ff627@x.com	$2b$12$klZkmD6Wj7or7TDHOPXBPOAogwarTjZsV5PYR2ONwgm0bvabVKcIG	Sam Agent	\N	salesperson	beginner	f	2026-06-07 19:11:15.151654+03	2026-06-07 19:11:16.238312+03	2aa5f809-d322-4d3d-85c6-1aeeac22539e
0de77f38-a75a-45e9-b270-45ea10eb00ad	m_592cd5df@x.com	$2b$12$Gp5t1LieVPAumq/3cQAAG.MXAGFLqVRXOlld7AUMPx3Hbg84lZ/XC	Mgr	Beta 592cd5df	manager	beginner	t	2026-06-07 19:12:53.54186+03	2026-06-07 19:12:53.54186+03	50c41bbb-26cc-46b8-9c06-20a8cd91b2f4
ad11f0b9-1b46-4569-8d0a-42c217f4d056	m_f7d91306@x.com	$2b$12$RODQOK1UNcR/aAbtZEzRcu/4dzanP0zLjTE50fh9wQAesJALIV2pe	Manager	Beta f7d91306	manager	beginner	t	2026-06-07 19:13:43.993885+03	2026-06-07 19:13:43.993885+03	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3
365f96ec-df61-4b26-bb30-5486858b0dcf	ag0_f7d91306@x.com	$2b$12$4T91Batqetde9k781tqZGeNnk5uf2a3Pct3JYIiA38Jd3Y33TwHju	Agent 0	\N	salesperson	beginner	t	2026-06-07 19:13:44.948072+03	2026-06-07 19:13:44.948072+03	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3
1cf3cfdb-d9ac-4d27-b56d-832e099b2781	ag1_f7d91306@x.com	$2b$12$mirWhEVsQpEMBua9HSdo.e.LtJgXLJxqpTUAYrnq5AfVCDidSe46S	Agent 1	\N	salesperson	beginner	t	2026-06-07 19:13:45.89101+03	2026-06-07 19:13:45.89101+03	1a5eb939-4df3-4358-8dc0-10b8dd33d2e3
c484ccea-ba5f-42ff-94d5-2b07274e86d3	bakr@hegazi.com	$2b$12$L14LAz3EZHAjjAB47WkMleOAok46mOb1.kyVvyRzjiy1ei7xnSJ3.	Bakr	Hegazi INC.	manager	beginner	t	2026-06-07 19:15:00.302097+03	2026-06-07 19:15:00.302097+03	7920d481-911e-43a8-8778-db740e193687
ec3fb8d3-1217-40b1-9233-263937c19f9b	mgr_59da9345@example.com	$2b$12$bhSp2eYO2NlxA3uJP2RVX.HWTL4TEUqNVnAFZulToLsQf4tW9L842	Jane Manager	Acme 59da9345	manager	beginner	t	2026-06-07 19:15:04.021433+03	2026-06-07 19:15:04.021433+03	caaad1c6-ffa3-43e1-8ed7-949d08c486a7
3723d638-6f27-4f92-9164-4ff388bf0eef	a1_59da9345@x.com	$2b$12$dED3ZAMAYMqTmgiOIzsfieHi.UzsQiL5au1kmhqRi0YqLvd38fgvu	Sam Agent	\N	salesperson	beginner	f	2026-06-07 19:15:05.012425+03	2026-06-07 19:15:06.006633+03	caaad1c6-ffa3-43e1-8ed7-949d08c486a7
b5baa5d9-7b71-4792-9cbd-3a360368212b	b0_59da9345@x.com	$2b$12$QTDAA7PP65WCSQBiHqz6EukD984H2u7CT7HSL9YpsLpO7LN0RQn5m	B 0	\N	salesperson	beginner	t	2026-06-07 19:15:06.082712+03	2026-06-07 19:15:06.082712+03	caaad1c6-ffa3-43e1-8ed7-949d08c486a7
348abbd9-92ab-4e7c-89ed-3b69de4df74a	mgr_91266b9f@example.com	$2b$12$qU5LEqKfYfk0rFUFcF5.fu06Fy1m5oj8W/MP/vivE7Bg9fHIpmmbS	Jane Manager	Acme 91266b9f	manager	beginner	t	2026-06-07 19:16:06.272556+03	2026-06-07 19:16:06.272556+03	25709324-afa8-415b-a277-3a00eaf144aa
2c878a76-c715-41b2-87fa-6fc7cc862f88	a1_91266b9f@x.com	$2b$12$dfcJm7J/P1LtJKccCLLbQe0WHRpLMFPCplaGONw6WuWuPM4hrOKKO	Sam Agent	\N	salesperson	beginner	f	2026-06-07 19:16:07.30917+03	2026-06-07 19:16:08.249253+03	25709324-afa8-415b-a277-3a00eaf144aa
399b588b-462b-4352-a691-30ae3fc3a6ad	mgr_da9e7b0b@example.com	$2b$12$YfWTJcWnJFI09eYBG03lcORVEtitif3c.lx7OHQ5jNpJQSStkYkce	Bob Manager	Beta da9e7b0b	manager	beginner	t	2026-06-07 19:16:08.287317+03	2026-06-07 19:16:08.287317+03	5276805f-82f4-4c23-9a87-6b1b30c4f3f8
c8046705-322b-4c07-a682-c6b3a5c14f89	b0_da9e7b0b@x.com	$2b$12$3cWL/aQVw6O6k/9SKe7BtObbeBcFKDTl5hS4ehXUtmoNVSap3K9eG	B 0	\N	salesperson	beginner	t	2026-06-07 19:16:09.050211+03	2026-06-07 19:16:09.050211+03	5276805f-82f4-4c23-9a87-6b1b30c4f3f8
0140a01c-2d01-4bf7-b146-8556b6309b34	b1_da9e7b0b@x.com	$2b$12$al6jgTiMhU0G.U2BxMVQwO3579GB4SPa9H3sdaHGpgxj4GpnEItf.	B 1	\N	salesperson	beginner	t	2026-06-07 19:16:09.788341+03	2026-06-07 19:16:09.788341+03	5276805f-82f4-4c23-9a87-6b1b30c4f3f8
fc567435-4933-4b6f-842c-0c8da1d11a8a	bakr@test.com	$2b$12$F2F26evObnFLOEo9w5aHwOq1yWXlRdzImkctO1VCubzyGp0LC0NZq	Abubakr		salesperson	beginner	t	2026-06-07 19:31:34.024413+03	2026-06-07 19:31:34.024413+03	\N
4e45d795-ae16-4cf2-b9c6-ba173335711a	abubakr@hegazi.com	$2b$12$wk0ct0EYZrizboYijrF9nu60ELkG1Xv2Urdgw3d9yrA0AWYuJkv5e	Abubakr	MIU	manager	beginner	t	2026-06-07 19:50:57.027234+03	2026-06-07 19:50:57.027234+03	10842b3e-ef84-4bca-9282-eb40d29be97d
\.


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: audit_logs audit_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_logs
    ADD CONSTRAINT audit_logs_pkey PRIMARY KEY (id);


--
-- Name: checkpoints checkpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (id);


--
-- Name: companies companies_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (id);


--
-- Name: companies companies_slug_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_slug_key UNIQUE (slug);


--
-- Name: emotion_logs emotion_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.emotion_logs
    ADD CONSTRAINT emotion_logs_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (id);


--
-- Name: personas personas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.personas
    ADD CONSTRAINT personas_pkey PRIMARY KEY (id);


--
-- Name: plans plans_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plans
    ADD CONSTRAINT plans_pkey PRIMARY KEY (name);


--
-- Name: seat_invites seat_invites_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_invites
    ADD CONSTRAINT seat_invites_pkey PRIMARY KEY (id);


--
-- Name: seat_invites seat_invites_token_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_invites
    ADD CONSTRAINT seat_invites_token_key UNIQUE (token);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: subscriptions subscriptions_company_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_company_id_key UNIQUE (company_id);


--
-- Name: subscriptions subscriptions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_pkey PRIMARY KEY (id);


--
-- Name: usage_periods uq_usage_company_period; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usage_periods
    ADD CONSTRAINT uq_usage_company_period UNIQUE (company_id, period_start);


--
-- Name: usage_periods usage_periods_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usage_periods
    ADD CONSTRAINT usage_periods_pkey PRIMARY KEY (id);


--
-- Name: user_stats user_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_stats
    ADD CONSTRAINT user_stats_pkey PRIMARY KEY (id);


--
-- Name: user_stats user_stats_user_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_stats
    ADD CONSTRAINT user_stats_user_id_key UNIQUE (user_id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: idx_checkpoints_session_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_checkpoints_session_id ON public.checkpoints USING btree (session_id);


--
-- Name: idx_emotion_logs_session_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_emotion_logs_session_id ON public.emotion_logs USING btree (session_id);


--
-- Name: idx_messages_session_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_messages_session_id ON public.messages USING btree (session_id);


--
-- Name: idx_messages_turn_number; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_messages_turn_number ON public.messages USING btree (session_id, turn_number);


--
-- Name: idx_sessions_started_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_started_at ON public.sessions USING btree (started_at DESC);


--
-- Name: idx_sessions_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_status ON public.sessions USING btree (status);


--
-- Name: idx_sessions_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_user_id ON public.sessions USING btree (user_id);


--
-- Name: ix_audit_logs_company_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_audit_logs_company_id ON public.audit_logs USING btree (company_id);


--
-- Name: ix_companies_slug; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_companies_slug ON public.companies USING btree (slug);


--
-- Name: ix_seat_invites_company_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_seat_invites_company_id ON public.seat_invites USING btree (company_id);


--
-- Name: ix_seat_invites_token; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_seat_invites_token ON public.seat_invites USING btree (token);


--
-- Name: ix_users_company_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_users_company_id ON public.users USING btree (company_id);


--
-- Name: audit_logs audit_logs_actor_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_logs
    ADD CONSTRAINT audit_logs_actor_user_id_fkey FOREIGN KEY (actor_user_id) REFERENCES public.users(id);


--
-- Name: audit_logs audit_logs_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_logs
    ADD CONSTRAINT audit_logs_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


--
-- Name: checkpoints checkpoints_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


--
-- Name: emotion_logs emotion_logs_message_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.emotion_logs
    ADD CONSTRAINT emotion_logs_message_id_fkey FOREIGN KEY (message_id) REFERENCES public.messages(id) ON DELETE CASCADE;


--
-- Name: emotion_logs emotion_logs_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.emotion_logs
    ADD CONSTRAINT emotion_logs_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


--
-- Name: users fk_users_company_id_companies; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT fk_users_company_id_companies FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


--
-- Name: messages messages_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


--
-- Name: seat_invites seat_invites_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_invites
    ADD CONSTRAINT seat_invites_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


--
-- Name: seat_invites seat_invites_invited_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_invites
    ADD CONSTRAINT seat_invites_invited_by_fkey FOREIGN KEY (invited_by) REFERENCES public.users(id);


--
-- Name: sessions sessions_persona_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_persona_id_fkey FOREIGN KEY (persona_id) REFERENCES public.personas(id);


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: subscriptions subscriptions_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


--
-- Name: subscriptions subscriptions_plan_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_plan_name_fkey FOREIGN KEY (plan_name) REFERENCES public.plans(name);


--
-- Name: usage_periods usage_periods_company_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usage_periods
    ADD CONSTRAINT usage_periods_company_id_fkey FOREIGN KEY (company_id) REFERENCES public.companies(id) ON DELETE CASCADE;


--
-- Name: user_stats user_stats_best_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_stats
    ADD CONSTRAINT user_stats_best_session_id_fkey FOREIGN KEY (best_session_id) REFERENCES public.sessions(id);


--
-- Name: user_stats user_stats_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_stats
    ADD CONSTRAINT user_stats_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict 7bR4RSJzoN5QL6B3sU6sbFlh7orIweMXFJwNdhmQnuA23a9SqX4s0gqb9hudugh

