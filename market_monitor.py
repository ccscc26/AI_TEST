from datetime import timezone, timedelta
utc_now = datetime.now(timezone.utc)
beijing_now = utc_now.astimezone(timezone(timedelta(hours=8)))
today = beijing_now.strftime("%Y-%m-%d")
