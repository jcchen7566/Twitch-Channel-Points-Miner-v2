import re

def cluster_title(title: str) -> str:
    """
    Groups titles into logical game regimes based on keyword analysis.
    This helps isolate signals for different types of content.
    """
    title_norm = title.lower()
    
    # 1. League of Legends (Strong competitive signal)
    if any(k in title_norm for k in ["lol", "英雄聯盟", "勝敗", "輸贏", "潘森", "這把"]):
        return "REGIME_LOL"
    
    # 2. Teamfight Tactics (TFT) (High variance, luck/skill mix)
    if any(k in title_norm for k in ["tft", "戰棋", "棋", "第幾"]):
        return "REGIME_TFT"
    
    # 3. Gambling/Luck games (Pure odds based)
    if any(k in title_norm for k in ["大富翁", "彈珠", "賭", "過關", "幾場"]):
        return "REGIME_LUCK"
    
    # 4. Variety/Chatting
    if any(k in title_norm for k in ["聊", "唱歌", "日常"]):
        return "REGIME_CHAT"
        
    return "REGIME_VARIETY"

def get_event_regime(event) -> str:
    """
    Returns the composite key: STREAMER_GAME
    """
    game_regime = cluster_title(event.main_title)
    return f"{event.streamer}_{game_regime}"
