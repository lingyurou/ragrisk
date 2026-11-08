# -*- coding: utf-8 -*-
"""Fetch the real regulatory documents used by the BankKB-R case study.

Every unit in units.py must quote a verbatim substring of the extracted text of
the document it points at; verify_quotes() enforces this, so no unit text can be
invented.  Raw bytes + sha256 are kept in raw/ and manifest.json.
"""
import hashlib, json, os, re, sys, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

DOCS = [
    dict(id="deposit_insurance", title="存款保险条例", ref="国务院令第660号",
         issuer="国务院", level="行政法规",
         published="2015-02-17", effective="2015-05-01",
         url="https://www.gov.cn/gongbao/content/2015/content_2843768.htm"),
    dict(id="ncd_rule", title="大额存单管理暂行办法", ref="中国人民银行公告〔2015〕第13号",
         issuer="中国人民银行", level="部门规范性文件",
         published="2015-06-02", effective="2015-06-02",
         url="https://www.gov.cn/gongbao/content/2015/content_2953961.htm"),
    dict(id="amc_guideline", title="关于规范金融机构资产管理业务的指导意见", ref="银发〔2018〕106号",
         issuer="人民银行 银保监会 证监会 外汇局", level="部门规范性文件",
         published="2018-04-27", effective="2018-04-27",
         url="https://www.gov.cn/gongbao/content/2018/content_5323101.htm"),
    dict(id="wealth_mgmt", title="商业银行理财业务监督管理办法", ref="银保监会令2018年第6号",
         issuer="银保监会", level="部门规章",
         published="2018-09-28", effective="2018-09-28",
         url="https://www.gov.cn/gongbao/content/2018/content_5350052.htm"),
    dict(id="structured_deposit", title="关于进一步规范商业银行结构性存款业务的通知", ref="银保监办发〔2019〕204号",
         issuer="银保监会办公厅", level="部门规范性文件",
         published="2019-10-18", effective="2019-10-18",
         url="https://www.gov.cn/zhengce/zhengceku/2019-12/03/content_5458085.htm"),
    dict(id="internet_deposit", title="关于规范商业银行通过互联网开展个人存款业务有关事项的通知", ref="银保监办发〔2021〕9号",
         issuer="银保监会办公厅 人民银行办公厅", level="部门规范性文件",
         published="2021-01-13", effective="2021-01-13",
         url="https://www.gov.cn/zhengce/zhengceku/2021-01/15/content_5580271.htm"),
    dict(id="loan_apr", title="中国人民银行公告〔2021〕第3号（明示贷款年化利率）", ref="中国人民银行公告〔2021〕第3号",
         issuer="中国人民银行", level="部门规范性文件",
         published="2021-03-12", effective="2021-03-12",
         url="https://www.gov.cn/zhengce/zhengceku/2021-03/31/content_5597156.htm"),
    dict(id="credit_card", title="关于进一步促进信用卡业务规范健康发展的通知", ref="银保监规〔2022〕13号",
         issuer="银保监会 人民银行", level="部门规范性文件",
         published="2022-07-07", effective="2022-07-07",
         url="https://jrj.sh.gov.cn/YWTBZCCX166/20220708/fed715163b37455397a37eb5a9938827.html",
         mirror="上海市地方金融监管局转载全文（nfra 原文为 JS 渲染）"),
    dict(id="pension_rule", title="个人养老金实施办法", ref="人社部发〔2022〕70号",
         issuer="人社部 财政部 税务总局 银保监会 证监会", level="部门规范性文件",
         published="2022-10-26", effective="2022-10-26",
         url="https://app.www.gov.cn/govdata/gov/202211/06/493994/article.html",
         mirror="国务院客户端全文（mohrss 原始链接有反爬）"),
    dict(id="pension_expand", title="关于全面实施个人养老金制度的通知", ref="人社部发〔2024〕87号",
         issuer="人社部 财政部 税务总局 金融监管总局 证监会", level="部门规范性文件",
         published="2024-12-10", effective="2024-12-15",
         url="http://beijing.chinatax.gov.cn/bjswj/c104581/202412/a23625d7b33a487ea05ed0f5d9cf463c.shtml",
         mirror="北京市税务局转载全文（mohrss 原始链接有反爬）"),
    dict(id="pension_leaflet", title="个人养老金政策宣传折页", ref="（官方宣传物料，宁夏社保局）",
         issuer="宁夏回族自治区社会保险事业管理局", level="官方宣传物料",
         published="2026-09-07", effective="2026-09-07",
         url="http://si.nx.gov.cn/xxgkai/xxgk/ggl_1/202609/t20260907_5336696.html"),
]

TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def fetch(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Language": "zh-CN,zh;q=0.9"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"fetch failed {url}: {last}")


def to_text(b):
    for enc in ("utf-8", "gb18030"):
        try:
            s = b.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        s = b.decode("utf-8", "ignore")
    s = TAG.sub(" ", s)
    s = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
          .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&ldquo;", "\u201c").replace("&rdquo;", "\u201d").replace("&mdash;", "\u2014").replace("&ldquo;", "\u201c"))
    s = re.sub(r"[ \t\u3000]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest = []
    for d in DOCS:
        path_html = os.path.join(RAW, d["id"] + ".html")
        path_txt = os.path.join(RAW, d["id"] + ".txt")
        if os.path.exists(path_txt) and os.path.getsize(path_txt) > 2000:
            b = open(path_html, "rb").read()
            print("cached", d["id"], len(b))
        else:
            b = fetch(d["url"])
            open(path_html, "wb").write(b)
            txt = to_text(b)
            open(path_txt, "w", encoding="utf-8").write(txt)
            print("fetched", d["id"], len(b), "->", len(txt))
        txt = open(path_txt, encoding="utf-8").read()
        m = dict(d)
        m["sha256"] = hashlib.sha256(open(path_html, "rb").read()).hexdigest()
        m["bytes"] = os.path.getsize(path_html)
        m["text_chars"] = len(txt)
        m["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest.append(m)
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
