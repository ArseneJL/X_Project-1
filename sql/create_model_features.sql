create table if not exists public.model_features (
    id bigserial primary key,

    symbol text not null,
    trade_date date not null,

    close numeric,
    volume numeric,

    return_1d numeric,
    return_5d numeric,
    return_20d numeric,

    sma_5 numeric,
    sma_20 numeric,
    sma_60 numeric,

    price_vs_sma20 numeric,
    volume_change_5d numeric,
    volatility_20 numeric,

    has_earnings_nearby boolean default false,
    has_dividend_nearby boolean default false,

    target_return_5d numeric,
    target_up_5d integer,

    created_at timestamptz default now(),
    updated_at timestamptz default now(),

    constraint model_features_symbol_date_unique unique (symbol, trade_date)
);
