-- Supabase / PostgreSQL schema for Stock Data Platform
-- Step 1: run this in Supabase SQL Editor

create extension if not exists "pgcrypto";

-- 1) Asset master table: stocks and indices
create table if not exists public.assets (
    id uuid primary key default gen_random_uuid(),
    symbol text not null unique,
    display_name text,
    asset_type text not null check (asset_type in ('stock', 'index', 'etf', 'future', 'crypto')),
    market text,
    sector text,
    currency text default 'USD',
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- 2) Daily OHLCV table
create table if not exists public.daily_prices (
    id bigserial primary key,
    symbol text not null references public.assets(symbol) on update cascade,
    trade_date date not null,
    open numeric(18,6),
    high numeric(18,6),
    low numeric(18,6),
    close numeric(18,6),
    volume bigint,
    source text default 'csv',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint daily_prices_symbol_date_unique unique(symbol, trade_date)
);

-- 3) Earnings events table
create table if not exists public.earnings (
    id bigserial primary key,
    symbol text not null references public.assets(symbol) on update cascade,
    earnings_date date not null,
    eps_reported numeric(18,6),
    eps_estimate numeric(18,6),
    surprise numeric(18,6),
    source text default 'csv',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint earnings_symbol_date_unique unique(symbol, earnings_date)
);

-- 4) Dividends table
create table if not exists public.dividends (
    id bigserial primary key,
    symbol text not null references public.assets(symbol) on update cascade,
    dividend_date date not null,
    dividend numeric(18,6) not null,
    source text default 'csv',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint dividends_symbol_date_unique unique(symbol, dividend_date)
);

-- 5) Pipeline logs table
create table if not exists public.pipeline_logs (
    id bigserial primary key,
    pipeline_name text not null,
    symbol text,
    start_date date,
    end_date date,
    rows_processed integer default 0,
    status text not null check (status in ('success', 'failed', 'partial')),
    error_message text,
    started_at timestamptz not null default now(),
    finished_at timestamptz
);

-- Indexes for dashboard query performance
create index if not exists idx_daily_prices_symbol_date on public.daily_prices(symbol, trade_date);
create index if not exists idx_earnings_symbol_date on public.earnings(symbol, earnings_date);
create index if not exists idx_dividends_symbol_date on public.dividends(symbol, dividend_date);

-- Basic seed assets for current project
insert into public.assets (symbol, display_name, asset_type, market, currency)
values
    ('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 'USD'),
    ('AMZN', 'Amazon.com Inc.', 'stock', 'NASDAQ', 'USD'),
    ('GOOGL', 'Alphabet Inc.', 'stock', 'NASDAQ', 'USD'),
    ('META', 'Meta Platforms Inc.', 'stock', 'NASDAQ', 'USD'),
    ('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 'USD'),
    ('NVDA', 'NVIDIA Corporation', 'stock', 'NASDAQ', 'USD'),
    ('TSLA', 'Tesla Inc.', 'stock', 'NASDAQ', 'USD'),
    ('S&P500', 'S&P 500 Index', 'index', 'US', 'USD'),
    ('NASDAQ100', 'NASDAQ 100 Index', 'index', 'US', 'USD')
on conflict (symbol) do update set
    display_name = excluded.display_name,
    asset_type = excluded.asset_type,
    market = excluded.market,
    currency = excluded.currency,
    updated_at = now();
