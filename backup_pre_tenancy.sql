--
-- PostgreSQL database dump
--

\restrict SnUcW7atQqVrOKtzr7dm5ynO7OYEYdHQvA6mSfxtMiQT3I8oFesKEILhcLSLNAB

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
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Data for Name: checkpoints; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.checkpoints (id, session_id, turn_start, turn_end, summary, key_points, customer_preferences, objections_raised, created_at) FROM stdin;
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
-- Data for Name: sessions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sessions (id, user_id, persona_id, status, difficulty, started_at, ended_at, duration_seconds, overall_score, communication_score, product_knowledge_score, objection_handling_score, rapport_score, closing_score, strengths, weaknesses, recommendations, turn_count, created_at) FROM stdin;
3a255d2c-796c-4a5b-b7b2-9144d10c91bc	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	rushed_customer	completed	medium	2026-02-06 22:26:30.749089+02	2026-02-06 22:27:40.238387+02	69	\N	\N	\N	\N	\N	\N	[]	[]	[]	0	2026-02-06 22:26:30.749089+02
7222ce99-6d47-447a-99a3-d430b6c03b14	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	first_time_buyer	active	easy	2026-02-06 22:28:00.001121+02	\N	\N	\N	\N	\N	\N	\N	\N	[]	[]	[]	0	2026-02-06 22:28:00.001121+02
\.


--
-- Data for Name: user_stats; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.user_stats (id, user_id, total_sessions, completed_sessions, total_training_minutes, avg_overall_score, avg_communication_score, avg_product_knowledge_score, avg_objection_handling_score, avg_rapport_score, avg_closing_score, best_session_id, best_score, current_streak, longest_streak, last_session_date, updated_at) FROM stdin;
f69c855a-c176-404c-80d0-26c252a151cc	3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	0	0	0	\N	\N	\N	\N	\N	\N	\N	\N	0	0	\N	2026-02-06 22:26:18.343672+02
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, email, password_hash, full_name, company, role, experience_level, is_active, created_at, updated_at) FROM stdin;
3eb0c118-1f7e-4183-a8e1-142af9fbc1f9	a@a.com	$2b$12$zzWsT21c/yQmMFbM6DJKuetEuf8APngWa3ZA05eFePqUWFg8DlkSC	Abubakr	A	salesperson	beginner	t	2026-02-06 22:26:18.06929+02	2026-02-06 22:26:18.06929+02
\.


--
-- Name: checkpoints checkpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (id);


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
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


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
-- Name: messages messages_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE CASCADE;


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

\unrestrict SnUcW7atQqVrOKtzr7dm5ynO7OYEYdHQvA6mSfxtMiQT3I8oFesKEILhcLSLNAB

