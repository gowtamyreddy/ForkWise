"""Token pricing table (USD per 1M tokens) for cost attribution.

Prices are list prices a user can override; local/self-hosted models are
priced 0. A span only gets a dollar cost when it reports a model that is
in this table plus token counts. Edit freely: this is configuration, not
a claim about what any vendor charges today.
"""

PRICES_PER_MILLION = {
    # model: (input_usd_per_million, output_usd_per_million)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "llama-3.1-8b-local": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def span_cost(model, tokens_in, tokens_out, extra_cost_usd=0.0):
    """Dollar cost of one span. Unknown models cost 0 and are flagged by
    the caller via `priced=False` in the return tuple."""
    if model and model in PRICES_PER_MILLION:
        pin, pout = PRICES_PER_MILLION[model]
        return (
            (tokens_in or 0) * pin / 1_000_000
            + (tokens_out or 0) * pout / 1_000_000
            + (extra_cost_usd or 0.0),
            True,
        )
    return (extra_cost_usd or 0.0), bool(extra_cost_usd)
