"""Column layouts of the iPinYou log files."""

# Original iPinYou impression logs (schema.txt in wnzhang/make-ipinyou-data), no header.
RAW_IMP_COLUMNS = [
    "bidid", "timestamp", "logtype", "ipinyouid", "useragent", "ip", "region", "city",
    "adexchange", "domain", "url", "urlid", "slotid", "slotwidth", "slotheight",
    "slotvisibility", "slotformat", "slotprice", "creative", "bidprice", "payprice",
    "keypage", "advertiser", "usertag",
]  # fmt: skip

# train.log.txt / test.log.txt from make-ipinyou-data: click, weekday and hour prepended.
MAKE_IPINYOU_COLUMNS = ["click", "weekday", "hour", *RAW_IMP_COLUMNS]
