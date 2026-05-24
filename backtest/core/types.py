from dataclasses import dataclass, field


@dataclass
class OutcomeRow:
    timestamp: str
    event_id: str
    streamer: str
    main_title: str
    outcome_id: str
    odds: float
    users_pct: float
    points_pct: float
    users: float
    points: float
    total_users: float
    total_points: float
    hw: float
    ht: float
    rw: float
    rt: float


@dataclass
class Event:
    timestamp: str
    event_id: str
    streamer: str
    main_title: str
    outcomes: list[OutcomeRow]
    winner_outcome_id: str
    selected_outcome_id: str
    result_type: str
    label_known: bool


@dataclass
class BacktestConfig:
    min_bet: float = 20000.0
    max_bet: float = 120000.0


@dataclass
class SliceStats:
    bet: float = 0.0
    gain: float = 0.0
    wins: int = 0
    losses: int = 0
    bets: int = 0


@dataclass
class BacktestResult:
    name: str
    bets: int
    total_bet: float
    total_gain: float
    roi: float
    win_rate: float
    avg_bet: float
    max_drawdown: float
    skipped_unlabeled_events: int
    by_streamer: dict[str, SliceStats] = field(default_factory=dict)
    by_regime: dict[str, SliceStats] = field(default_factory=dict)
    by_liquidity: dict[str, SliceStats] = field(default_factory=dict)
