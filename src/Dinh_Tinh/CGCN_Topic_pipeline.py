# -*- coding: utf-8 -*-
"""
CGCN TOPIC PIPELINE — 1 file duy nhất, chạy trên máy bạn (local).

Không có chủ đề/từ điển nào định trước — mọi chủ đề và cách gộp chủ đề đều
do thuật toán tự tìm từ dữ liệu của bạn. Có 2 ENGINE, chọn ở phần CẤU HÌNH:

  ENGINE = "tfidf_nmf"  (mặc định)
    TF-IDF + NMF, tự chọn số chủ đề K bằng silhouette score. Chỉ cần
    scikit-learn — chạy vài giây, không cần tải model, không cần GPU.

  ENGINE = "bertopic"
    Sentence embedding thật (mặc định: AITeamVN/Vietnamese_Embedding — xem lựa
    chọn khác, điểm benchmark và giải thích quan hệ với PhoBERT ngay tại
    BERTOPIC_MODEL_NAME ở phần CẤU HÌNH) + BERTopic
    (UMAP + HDBSCAN) — HDBSCAN tự quyết định SỐ CHỦ ĐỀ theo mật độ dữ liệu,
    không cần quét K như NMF. Cần cài thêm sentence-transformers, bertopic,
    umap-learn, hdbscan, torch — và TẢI MODEL (500MB-1GB tùy model) ở lần
    chạy đầu, nên cần mạng lúc đó. Đây là engine CHƯA được chạy thử trong
    sandbox của tôi (sandbox không có internet tới PyPI/HuggingFace, kể cả để
    tự tải model kiểm chứng giúp bạn) — đã kiểm tra cú pháp (py_compile)
    nhưng bạn là người đầu tiên chạy thật; nếu lỗi, gửi lại thông báo lỗi để
    tôi sửa, hoặc tạm dùng ENGINE="tfidf_nmf".

CÁC BƯỚC CHUNG CHO CẢ 2 ENGINE:
  1. Phân đoạn văn bản thành "tài liệu" (3 chế độ: luật / bài báo-whitepaper /
     thư mục nhiều file).
  2. Tách từ tiếng Việt bằng khớp dài nhất (maximum matching) trên từ điển
     ~70.000 từ + lọc từ dừng bằng danh sách ~1.900 từ — cả hai đều là tài
     nguyên công khai, KHÔNG phải từ điển chủ đề. Nếu chưa có 2 file này cạnh
     script, script TỰ TẢI VỀ 1 lần từ GitHub (cần internet).
  3. Phát hiện chủ đề (bậc 2) — theo ENGINE đã chọn ở trên.
  4. Kiểm tra độ vững: so 3 (hoặc 2, với bertopic) cách phân cụm độc lập
     bằng ARI.
  5. MDS — vẽ hình học nội dung, tô màu theo chủ đề tự phát hiện.
  6. Mã hóa trục & chọn lọc: GỘP TỰ ĐỘNG chủ đề bậc 2 -> khía cạnh bậc 3
     bằng phân cụm phả hệ (hierarchical clustering) trên vector chủ đề —
     không phải tôi tự gộp bằng mắt. Tính tỷ trọng % + trích bằng chứng đại
     diện cho mỗi khía cạnh.

CÁI GÌ KHÔNG TỰ ĐỘNG HÓA ĐƯỢC (và không nên giả vờ tự làm), dù dùng engine
nào:
  - Đặt TÊN có nghĩa pháp lý cho từng chủ đề/khía cạnh — thuật toán chỉ đưa
    ra từ khóa + văn bản đại diện, không có NLP nào (kể cả PhoBERT) tự sinh
    ra tên có nghĩa pháp lý.
  - Gán vai trò trong mô hình paradigm (điều kiện nhân quả / hiện tượng cốt
    lõi / điều kiện bối cảnh / điều kiện can thiệp / chiến lược / hệ quả) —
    đây là suy luận lý thuyết của người nghiên cứu (Nelson 2020 gọi là
    "tinh chỉnh mẫu bằng diễn giải lý thuyết", bước riêng sau bước phát hiện
    mẫu bằng thuật toán).
  Cả hai để TRỐNG có chủ đích trong các CSV xuất ra — bạn tự điền.

CÀI ĐẶT:
  ENGINE="tfidf_nmf" : pip install scikit-learn scipy numpy pandas matplotlib
  ENGINE="bertopic"  : thêm pip install sentence-transformers bertopic
                        umap-learn hdbscan torch
  LDA (tùy chọn, chỉ ENGINE="tfidf_nmf"): thêm pip install gensim — dùng làm 1
                        thuật toán đối chứng độc lập thêm ở bước [6] "Độ vững"
                        (xem giải thích tại chỗ dùng trong code). Không cài
                        cũng không sao, script tự bỏ qua.
"""
import os
import re
import glob
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix

# =============================================================================
# 0. CẤU HÌNH — SỬA CÁC DÒNG NÀY CHO CORPUS CỦA BẠN
# =============================================================================
SRC = r"D:\vbpl-crawler\Dataset_Dinh_tinh"      # đường dẫn văn bản (MODE="legal"/"generic") HOẶC
                            # đường dẫn thư mục (MODE="files")
MODE = "files"              # "legal"   : 1 file văn bản luật, tách theo "Điều N."
                             # "generic" : 1 file bài báo/whitepaper, tách theo
                             #             đề mục (số thứ tự / I. II. / Chương...),
                             #             rơi về tách đoạn nếu không đủ đề mục
                             # "files"   : 1 thư mục, mỗi file .txt = 1 tài liệu
                             #             (dùng khi gộp nhiều luật/nhiều bài báo)
ENGINE = "tfidf_nmf"        # "tfidf_nmf" (nhẹ, chạy ngay) | "bertopic" (embedding thật, xem BERTOPIC_MODEL_NAME)
N_AGGREGATE = None          # số khía cạnh bậc 3 mong muốn ở bước 6; None = tự chọn
FORCE_K =  7          # (chỉ áp dụng ENGINE="tfidf_nmf") None = tự quét K bằng
                             # silhouette như bình thường. Đặt 1 số nguyên (vd 8) để
                             # BỎ QUA quét, dùng đúng K đó — hữu ích khi silhouette
                             # tăng đơn điệu tới sát số tài liệu (dấu hiệu công thức
                             # đang tiến về lời giải suy biến "mỗi tài liệu 1 cụm" chứ
                             # không phải K tối ưu thật) và bạn muốn tự chọn K theo số
                             # lượng "chủ đề bậc 2" hợp lý để mã hóa trục, thay vì tin
                             # tuyệt đối vào con số thuật toán chọn.
K_SCAN_REPORT = []          # (chỉ áp dụng ENGINE="tfidf_nmf") để trống [] = bỏ qua
                             # (mặc định). Đặt 1 danh sách K muốn so sánh nhanh, vd
                             # [3, 5, 6, 7, 8], để in ra 1 bảng silhouette + %chủ đề
                             # lớn nhất + ARI cho TỪNG K đó TRƯỚC khi chạy phân tích
                             # đầy đủ — giúp chọn giá trị để đặt FORCE_K mà không phải
                             # tự sửa FORCE_K rồi chạy lại nhiều lần bằng tay.

BERTOPIC_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
    # ^ Đổi dòng trên thành 1 trong các lựa chọn dưới đây tùy máy/mục tiêu.
    # Điểm VN-MTEB lấy từ benchmark độc lập (arXiv:2507.21500 / ACL 2026.findings-
    # eacl.86) — tôi KHÔNG tự tải/chạy được các model này để đo (huggingface.co bị
    # chặn trong sandbox này), điểm số là số công bố trong paper, không phải tôi đo.
    #
    #   "AITeamVN/Vietnamese_Embedding"          568M | VN-MTEB 63.34 | ĐANG CHỌN.
    #                                             Fine-tune TỪ BAAI/bge-m3 (không phải
    #                                             từ PhoBERT — xem giải thích PhoBERT
    #                                             bên dưới). Model card tự báo cáo
    #                                             MRR@10=0.818 trên tập truy vấn pháp
    #                                             luật Zalo Legal 2021 (không nằm
    #                                             trong VN-MTEB, do chính nhóm tác giả
    #                                             đo). Có bản v2 "Vietnamese_Embedding_v2"
    #                                             được 1 paper khác báo cáo vượt cả
    #                                             bge-m3/m-e5-large trên truy vấn pháp
    #                                             luật (ALQAC, ZaloLegalQA) — có thể thử
    #                                             đổi tên model thành bản v2 nếu muốn.
    #   "intfloat/multilingual-e5-large-instruct" 560M | VN-MTEB 67.99 (CAO NHẤT bảng,
    #                                             kể cả so với model 7B) | đa ngôn ngữ
    #   "BAAI/bge-m3"                             568M | VN-MTEB 64.90 | chính là model
    #                                             gốc mà Vietnamese_Embedding fine-tune
    #                                             lên — hỗ trợ văn bản dài (8192 token)
    #   "Alibaba-NLP/gte-multilingual-base"       305M | VN-MTEB 65.22 | nhẹ, nhanh
    #   "bkai-foundation-models/vietnamese-bi-encoder" 135M | VN-MTEB 54.89 | mặc định
    #                                             CŨ, fine-tune từ PhoBERT-base-v2, yếu
    #                                             nhất trong nhóm trên theo VN-MTEB
    #
    # Nếu máy KHÔNG có GPU: các model trên vẫn chạy được trên CPU với ~30 tài liệu
    # (chỉ chậm hơn, không cần đổi gì thêm) — script tự dùng CPU nếu không thấy GPU.
BERTOPIC_MIN_TOPIC_SIZE = None  # số tài liệu tối thiểu/chủ đề; None = tự chọn
                                 # heuristic theo cỡ corpus (xem bên dưới)

EMBED_DEVICE = "auto"     # "auto" (mặc định) | "cuda" | "cpu"
    # "auto": thử GPU trước; nếu GPU hết bộ nhớ (CUDA OOM) khi mã hóa, script TỰ
    # giảm batch_size rồi thử lại trên GPU, và nếu vẫn không đủ thì tự chuyển
    # toàn bộ phần còn lại sang CPU — không cần bạn sửa gì, không dừng chương
    # trình. (sentence-transformers không hỗ trợ chia 1 batch cho nửa GPU/nửa
    # CPU cùng lúc; "linh động" ở đây nghĩa là tự hạ cấp dần: GPU batch lớn ->
    # GPU batch nhỏ -> CPU, chứ không chạy song song 2 device.)
    # "cuda": ép luôn GPU (báo lỗi rõ nếu không có GPU thay vì âm thầm dùng CPU).
    # "cpu": ép luôn CPU (chậm hơn nhưng không bao giờ lo OOM).
EMBED_BATCH_SIZE = 8      # batch khởi đầu khi mã hóa trên GPU. GPU càng ít VRAM
                                 # (như 4GB) thì để càng nhỏ (4 hoặc 2); script sẽ tự
                                 # giảm tiếp nếu vẫn OOM. Trên CPU, batch_size không
                                 # ảnh hưởng nhiều tới việc có chạy được hay không.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR        # toàn bộ CSV/PNG xuất ra sẽ nằm cạnh script này

# =============================================================================
# 1. TỪ ĐIỂN TÁCH TỪ + STOPWORD — tự tải nếu chưa có, cache cạnh script
#    (dùng cho ENGINE="tfidf_nmf" và cho bước làm sạch từ khóa c-TF-IDF khi
#    ENGINE="bertopic"; bản thân embedding PhoBERT không cần bước lọc này)
# =============================================================================
WORDLIST_FILE = os.path.join(SCRIPT_DIR, "vietnamese-wordlist.txt")
STOPWORDS_FILE = os.path.join(SCRIPT_DIR, "vietnamese-stopwords.txt")
MAX_SPAN = 5  # số âm tiết dài nhất thử ghép thành 1 từ khi tách từ

_SOURCES = {
    WORDLIST_FILE: "https://raw.githubusercontent.com/duyet/vietnamese-wordlist/master/Viet74K.txt",
    STOPWORDS_FILE: "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt",
}

# Cụm pháp lý/CGCN không chắc có trong từ điển tiếng Việt tổng quát — thêm
# thẳng vào từ điển tách từ (không phải bước ghép cụm riêng) để cùng một
# thuật toán khớp dài nhất xử lý luôn, không cần cơ chế thứ hai.
_DOMAIN_TERMS = [
    "chuyển giao công nghệ", "sở hữu công nghiệp", "đổi mới sáng tạo",
    "bí quyết kỹ thuật", "công nghệ cao", "công nghệ tiên tiến",
    "công nghệ mới", "công nghệ sạch", "thị trường khoa học",
    "quản lý nhà nước", "dự án đầu tư", "bồi thường thiệt hại",
    "thẩm định giá", "quyền sở hữu", "việt nam", "khoa học công nghệ",
    "chuyển giao công nghệ trong nước", "vốn đầu tư công",
    # Tên quốc gia/khu vực — viet74k (từ điển gốc, thiên về từ vựng văn học/
    # phổ thông) KHÔNG có các cụm này, nên nếu thiếu thì bị tách rời thành 2
    # từ đơn nghĩa chung chung (vd "Trung Quốc" -> "trung" + "quốc"), phát
    # hiện được khi đọc kỹ output chạy thật trên corpus báo chí/thực tiễn:
    "trung quốc", "hàn quốc", "nhật bản", "hoa kỳ", "châu âu", "đông nam á",
    "liên minh châu âu",
    # "phát thải" tồn tại trong dữ liệu môi trường/CGCN xanh nhưng không có
    # sẵn trong từ điển gốc (chỉ có "phát" và "thải" riêng lẻ) -> cũng bị
    # tách rời, phát hiện tương tự.
    "phát thải",
]

def _clean_wordlist(raw_text: str) -> str:
    """Chuẩn hoá file từ điển thô (viet74k gốc có chữ hoa, vài mục lẫn số/
    ký hiệu) về chữ thường, NFC, chỉ giữ mục thuần chữ cái + khoảng trắng,
    rồi thêm _DOMAIN_TERMS. Không làm bước này thì nhiều mục hoa (vd "Việt
    Nam") sẽ không bao giờ khớp vì văn bản đưa vào preprocess() đã hạ chữ
    thường hết."""
    words = set()
    for line in raw_text.splitlines():
        w = line.strip()
        if not w:
            continue
        w = unicodedata.normalize("NFC", w.lower())
        if all(c.isalpha() or c == " " for c in w):
            words.add(w)
    for t in _DOMAIN_TERMS:
        words.add(unicodedata.normalize("NFC", t.lower()))
    return "\n".join(sorted(words)) + "\n"

def ensure_local_file(path):
    """Tải 1 lần từ GitHub nếu chưa có sẵn cạnh script; các lần chạy sau dùng
    bản cache, không cần mạng nữa. Không có mạng lúc chạy lần đầu -> in cảnh
    báo và rơi về danh sách rút gọn (không crash)."""
    if os.path.exists(path):
        return True
    url = _SOURCES[path]
    try:
        print(f"  [tải] {os.path.basename(path)} <- {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        if path == WORDLIST_FILE:
            raw = _clean_wordlist(raw)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
        return True
    except Exception as e:
        print(f"  [!] Không tải được {os.path.basename(path)} ({e}). "
              f"Tải tay file này và đặt cạnh script, hoặc kiểm tra mạng.")
        return False

ensure_local_file(WORDLIST_FILE)
ensure_local_file(STOPWORDS_FILE)

def load_wordset(path):
    if not os.path.exists(path):
        return None
    return {w.strip() for w in open(path, encoding="utf-8") if w.strip()}

VN_WORDS = load_wordset(WORDLIST_FILE)
# _DOMAIN_TERMS luôn được hợp nhất vào VN_WORDS TRONG BỘ NHỚ ở đây, dù file
# wordlist trên đĩa là mới tải hay đã cache từ lần chạy trước. Lý do: một khi
# vietnamese-wordlist.txt đã tồn tại cạnh script, ensure_local_file() ở trên
# sẽ KHÔNG tải/làm sạch lại (tránh tốn mạng mỗi lần chạy) -> nếu chỉ thêm vào
# _DOMAIN_TERMS mà không hợp nhất lại ở đây thì các cụm từ mới thêm (vd "trung
# quốc", "phát thải") sẽ KHÔNG có tác dụng cho tới khi xóa tay file cache. Hợp
# nhất ở đây đảm bảo mọi lần sửa _DOMAIN_TERMS đều có hiệu lực ngay, không cần
# xóa cache hay có mạng lại.
if VN_WORDS is not None:
    VN_WORDS |= {unicodedata.normalize("NFC", t.lower()) for t in _DOMAIN_TERMS}
STOPWORDS = load_wordset(STOPWORDS_FILE)
if STOPWORDS is None:
    STOPWORDS = {"là", "và", "của", "các", "được", "có", "cho", "theo", "trong",
                 "này", "những", "để", "về", "với", "khi", "tại", "từ", "hoặc",
                 "không", "đã", "sẽ", "phải", "như", "một", "đến"}
# rác cấu trúc riêng của thể loại văn bản luật (không phải từ dừng tiếng Việt
# nói chung) — vẫn lọc kể cả khi dùng danh sách rút gọn ở trên
STOPWORDS |= {"quy định", "điều", "khoản", "luật", "chương", "mục",
              "nghị định", "thông tư", "số", "ngày", "tháng", "năm"}
# rác đặc thù của corpus BÁO CHÍ/WEB (kho B — bài báo, hội thảo...), phát
# hiện khi đọc output chạy thật trên corpus 27 bài: học hàm/học vị viết tắt
# (đứng trước tên riêng trong lối viết báo chí, không mang nghĩa chủ đề) +
# rác kỹ thuật còn sót lại khi copy nội dung trang web vào .docx (ngoài
# https/www/html đã lọc bằng regex ở _normalize_text, đây là các TỪ literal
# còn lại trong nội dung, ví dụ "trang 3", "javascript" bị dính trong text)
STOPWORDS |= {"ts", "ths", "pgs", "gs", "th.s",
              "image", "javascript", "css", "script", "trang"}

def segment_words(sylls, wordset, max_span=MAX_SPAN):
    """Tách từ bằng khớp dài nhất (maximum matching): tại mỗi vị trí, thử
    ghép max_span âm tiết liền kề, giảm dần xuống 1; cụm nào có trong từ
    điển thì lấy làm 1 token. Dùng cho cả 2 engine — với bertopic, đây là
    bước thay thế underthesea.word_tokenize mà model PhoBERT/vietnamese-bi-
    encoder cần (model yêu cầu input đã tách từ, âm tiết trong 1 từ nối
    bằng '_'). Không có mô hình xác suất/ngữ cảnh nên vẫn kém hơn PhoBERT
    tokenizer thật, nhưng không hard-code theo chủ đề nào."""
    out, i, n = [], 0, len(sylls)
    while i < n:
        matched = False
        for span in range(min(max_span, n - i), 1, -1):
            cand = " ".join(sylls[i:i + span])
            if cand in wordset:
                out.append(cand.replace(" ", "_"))
                i += span
                matched = True
                break
        if not matched:
            out.append(sylls[i])
            i += 1
    return out

# Chuẩn hóa viết tắt PHỔ BIẾN, KHÔNG NHẬP NHẰNG trong văn bản pháp lý/tin tức
# tiếng Việt — mở rộng về dạng đầy đủ TRƯỚC khi tách từ, để "KH&CN" không vỡ
# thành 2 token vô nghĩa "kh"/"cn" (do dấu '&' bị coi là dấu câu và xóa ở
# bước sau). Đây là chuẩn hóa văn bản khách quan, áp dụng đồng đều cho mọi
# tài liệu — không phải gán nhãn/chủ đề. Chỉ liệt kê viết tắt không thể hiểu
# nhầm sang nghĩa khác trong ngữ cảnh CGCN/pháp luật/kinh tế.
_ABBREV_PATTERNS = [
    (re.compile(r"\bkh\s*&\s*cn\b|\bkh-cn\b|\bkhcn\b"), "khoa học công nghệ"),
    (re.compile(r"\bcgcn\b"), "chuyển giao công nghệ"),
    (re.compile(r"\bdnnn\b"), "doanh nghiệp nhà nước"),
    (re.compile(r"\bdn\b"), "doanh nghiệp"),
    (re.compile(r"\bshtt\b"), "sở hữu trí tuệ"),
    (re.compile(r"\bubnd\b"), "ủy ban nhân dân"),
]

def _normalize_text(s: str) -> str:
    """Bước chuẩn hóa dùng chung cho cả preprocess() và segment_for_embedding():
    hạ chữ thường, xóa URL (tin tức lấy từ web thường dính nguyên link, vỡ
    thành rác kiểu 'https'/'www'/'html' sau khi tách từ), rồi mở rộng viết
    tắt."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    for pat, repl in _ABBREV_PATTERNS:
        s = pat.sub(repl, s)
    return s

def preprocess(s: str) -> str:
    """Dùng cho ENGINE="tfidf_nmf": tách từ + BỎ stopword + BỎ số (TF-IDF
    chỉ cần từ nội dung)."""
    s = _normalize_text(s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\d+", " ", s)
    sylls = s.split()
    tokens = segment_words(sylls, VN_WORDS) if VN_WORDS else sylls
    return " ".join(t for t in tokens
                     if t.replace("_", " ") not in STOPWORDS and len(t) > 1)

def segment_for_embedding(s: str) -> str:
    """Dùng cho ENGINE="bertopic": chỉ tách từ (ghép âm tiết), GIỮ stopword
    và không bỏ số — embedding câu cần giữ ngữ cảnh đầy đủ hơn TF-IDF, chỉ
    bỏ dấu câu để khớp dài nhất hoạt động ổn định trên chuỗi âm tiết."""
    s = _normalize_text(s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    sylls = s.split()
    tokens = segment_words(sylls, VN_WORDS) if VN_WORDS else sylls
    return " ".join(tokens)

# =============================================================================
# 2. PHÂN ĐOẠN THÀNH "TÀI LIỆU" — 3 chế độ
# =============================================================================
def _ensure_docx_support():
    """Cần python-docx để đọc trực tiếp file .docx (không bắt bạn tự chuyển
    sang .txt trước). Thử tự cài 1 lần nếu chưa có — package này thuần
    Python, nhẹ, không cần biên dịch, an toàn để tự cài."""
    try:
        import docx  # noqa: F401
        return True
    except ImportError:
        pass
    print("  [cài] Chưa có python-docx — đang tự cài để đọc file .docx trực tiếp...")
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-docx"])
        import docx  # noqa: F401
        return True
    except Exception as e:
        print(f"  [!] Không tự cài được python-docx ({e}). Cài tay: pip install python-docx")
        return False

def read_text_file(path: str) -> str:
    """Đọc nội dung 1 file — hỗ trợ .txt (thử vài encoding phổ biến ở Việt
    Nam) và .docx trực tiếp (không cần bạn tự chuyển đổi trước)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        if not _ensure_docx_support():
            raise SystemExit(f"[LỖI] Không đọc được '{path}' — thiếu python-docx và không tự "
                              f"cài được. Cài tay: pip install python-docx")
        from docx import Document
        doc = Document(path)
        # ghép cả đoạn văn thường lẫn nội dung trong bảng (nhiều văn bản luật
        # có phụ lục/điều khoản trình bày dạng bảng)
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    for enc in ("utf-8", "utf-8-sig", "cp1258", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

HEADING_PATTERN = re.compile(
    r"^\s*(?:(?:\d{1,2}(?:\.\d{1,2}){0,2})\.?|[IVXLC]{1,4}\.|Phần\s+\d+|Chương\s+\d+)"
    r"[\s.:-]+\S.{0,80}$", re.MULTILINE,
)

def segment_legal(text: str) -> pd.DataFrame:
    pattern = re.compile(r"\nĐiều\s+(\d+)\.\s*([^\n]+)\n")
    matches = list(pattern.finditer(text))
    rows = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        rows.append({"id": f"Điều {m.group(1)}", "tieu_de": m.group(2).strip(),
                     "text": text[start:end].strip()})
    return pd.DataFrame(rows)

def segment_generic(text: str, min_words: int = 25) -> pd.DataFrame:
    headings = list(HEADING_PATTERN.finditer(text))
    rows = []
    if len(headings) >= 5:
        for i, m in enumerate(headings):
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if len(body.split()) >= min_words:
                rows.append({"id": f"Mục {i+1}", "tieu_de": m.group(0).strip()[:80], "text": body})
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= min_words]
        rows = [{"id": f"Đoạn {i+1}", "tieu_de": p[:60].replace("\n", " "), "text": p}
                for i, p in enumerate(paras)]
    return pd.DataFrame(rows)

def segment_files(dir_path: str) -> pd.DataFrame:
    if not os.path.isdir(dir_path):
        raise SystemExit(
            f"[LỖI] SRC='{dir_path}' không phải là thư mục hợp lệ (hoặc không tồn tại).\n"
            f"      Với MODE='files', SRC phải trỏ vào một THƯ MỤC, không phải 1 file."
        )
    # quét ĐỆ QUY (bao gồm thư mục con) — không chỉ file nằm trực tiếp trong
    # dir_path, vì corpus thường được tổ chức theo thư mục con đánh số. Đọc
    # được cả .txt lẫn .docx trực tiếp (xem read_text_file). Loại 2 file từ
    # điển/stopword tự tải (nếu SRC trùng/nằm trong SCRIPT_DIR) và file khóa
    # tạm "~$..." mà Word tạo ra khi file .docx đang mở — nếu không sẽ bị
    # lẫn vào corpus như "tài liệu" giả hoặc gây lỗi khi đọc.
    _skip_basenames = {os.path.basename(WORDLIST_FILE), os.path.basename(STOPWORDS_FILE)}
    _supported_exts = (".txt", ".docx")
    paths = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _supported_exts and fn not in _skip_basenames and not fn.startswith("~$"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    if not paths:
        all_exts = sorted({
            os.path.splitext(fn)[1].lower()
            for _, _, files in os.walk(dir_path) for fn in files if os.path.splitext(fn)[1]
        })
        raise SystemExit(
            f"[LỖI] Không tìm thấy file .txt hoặc .docx nào trong '{dir_path}' (đã quét cả thư "
            f"mục con).\n"
            f"      Các đuôi file thực tế có trong thư mục: {all_exts or '(thư mục trống)'}\n"
            f"      MODE='files' đọc được .txt và .docx. Nếu tài liệu của bạn là .doc/.pdf, cần\n"
            f"      chuyển sang .txt hoặc .docx trước (Word: File > Save As; hoặc pandoc)."
        )
    n_docx = sum(1 for p in paths if p.lower().endswith(".docx"))
    if n_docx:
        print(f"    Đọc {n_docx} file .docx trực tiếp (không cần chuyển sang .txt).")
    rows = []
    for i, path in enumerate(paths):
        text = read_text_file(path).strip()
        if not text:
            print(f"    [!] Bỏ qua file rỗng: {path}")
            continue
        rows.append({"id": f"Bài {i+1}", "tieu_de": os.path.relpath(path, dir_path),
                     "text": text})
    return pd.DataFrame(rows)

if MODE == "legal":
    if not os.path.isfile(SRC):
        raise SystemExit(f"[LỖI] SRC='{SRC}' không phải là file hợp lệ. Với MODE='legal', "
                          f"SRC phải là đường dẫn 1 file .txt hoặc .docx chứa văn bản luật.")
    df = segment_legal(read_text_file(SRC))
elif MODE == "generic":
    if not os.path.isfile(SRC):
        raise SystemExit(f"[LỖI] SRC='{SRC}' không phải là file hợp lệ. Với MODE='generic', "
                          f"SRC phải là đường dẫn 1 file .txt hoặc .docx bài báo/whitepaper.")
    df = segment_generic(read_text_file(SRC))
elif MODE == "files":
    df = segment_files(SRC)
else:
    raise ValueError("MODE phải là 'legal', 'generic' hoặc 'files'")

if df.empty:
    raise SystemExit(
        f"[LỖI] Phân đoạn ra 0 tài liệu (MODE='{MODE}', SRC='{SRC}').\n"
        f"      - MODE='legal': SRC phải là 1 file .txt/.docx có các dòng dạng 'Điều 1. ...',\n"
        f"        'Điều 2. ...' — kiểm tra file có đúng định dạng này không (vd file đã bị\n"
        f"        đổi encoding, hoặc không phải văn bản luật).\n"
        f"      - MODE='generic': SRC phải là 1 file .txt/.docx có đủ nội dung/đề mục.\n"
        f"      - MODE='files': SRC phải là 1 thư mục chứa file .txt/.docx (script đã quét cả\n"
        f"        thư mục con) — xem lỗi/cảnh báo cụ thể hơn ở phía trên nếu có."
    )

print(f"[1] Phân đoạn ('{MODE}'): {len(df)} tài liệu. Engine = {ENGINE}.")
if len(df) < 15:
    print(f"    CẢNH BÁO: chỉ {len(df)} tài liệu — quá ít để phân cụm/MDS/tương quan ổn định.")

# =============================================================================
# 3. GỘP TỰ ĐỘNG chủ đề bậc 2 -> khía cạnh bậc 3 + XUẤT WORKSHEET
#    (dùng chung cho cả 2 engine — chỉ khác nguồn topic_vectors/doc_confidence)
# =============================================================================
def aggregate_and_export(df, topic_names, topic_vectors, topic_keywords,
                          doc_confidence, file_suffix=""):
    """topic_vectors: mảng (K, D) — vector đại diện mỗi chủ đề bậc 2, dùng để
    phân cụm phả hệ (hierarchical clustering, cosine) thành khía cạnh bậc 3.
    Với NMF: D = kích thước từ vựng (hàng H). Với BERTopic: D = kích thước
    embedding (tâm cụm các văn bản thuộc chủ đề đó).
    doc_confidence: pd.Series (index trùng df.index) — tài liệu càng đại
    diện cho chủ đề của chính nó thì giá trị càng cao, dùng để chọn 'bằng
    chứng đại diện'.
    file_suffix: hậu tố tên file xuất ra, để 2 engine không ghi đè lên nhau
    (vd '' cho tfidf_nmf, '_bertopic' cho bertopic)."""
    K = len(topic_names)
    n_agg = N_AGGREGATE if N_AGGREGATE is not None else max(2, min(5, round(K / 2)))
    link = linkage(topic_vectors, method="average", metric="cosine")
    agg_id = fcluster(link, t=n_agg, criterion="maxclust")
    topic_to_agg = {topic_names[k]: f"Khía cạnh {agg_id[k]}" for k in range(K)}
    df["khia_canh_bac3"] = df["chu_de_bac2"].map(topic_to_agg)  # nhiễu/outlier -> NaN

    print(f"\n[Mã hóa trục] Gộp {K} chủ đề bậc 2 thành {n_agg} khía cạnh bậc 3 "
          f"(phân cụm phả hệ, cosine):")
    for agg in sorted(set(topic_to_agg.values())):
        members = [t for t, a in topic_to_agg.items() if a == agg]
        print(f"    {agg}: {', '.join(members)}")

    valid = df[df["khia_canh_bac3"].notna()]
    prop = valid["khia_canh_bac3"].value_counts(normalize=True).mul(100).round(1)

    def top_excerpts(agg, n=3, maxlen=140):
        sub = valid[valid["khia_canh_bac3"] == agg]
        conf = doc_confidence.reindex(sub.index).fillna(0)
        order = conf.sort_values(ascending=False).index[:n]
        return [f"{df.loc[i,'id']}: {df.loc[i,'text'][:maxlen].strip()}..." for i in order]

    worksheet_rows = []
    for agg in sorted(set(topic_to_agg.values())):
        members = [t for t, a in topic_to_agg.items() if a == agg]
        kw = []
        for t in members:
            kw += topic_keywords.get(t, [])[:6]
        worksheet_rows.append({
            "khia_canh_bac3": agg,
            "chu_de_bac2_gop": ", ".join(members),
            "ty_trong_%": prop.get(agg, 0.0),
            "tu_khoa_gop": ", ".join(dict.fromkeys(kw)),  # loại trùng, giữ thứ tự
            "bang_chung_dai_dien": " || ".join(top_excerpts(agg)),
            "ten_de_xuat": "",       # <- điền tay: tên khía cạnh có nghĩa pháp lý
            "vai_tro_paradigm": "",  # <- điền tay: nhân quả/cốt lõi/bối cảnh/can thiệp/chiến lược/hệ quả
            "ghi_chu_lap_luan": "",  # <- điền tay: vì sao xếp vào vai trò đó
        })
    worksheet = pd.DataFrame(worksheet_rows).sort_values("ty_trong_%", ascending=False)
    worksheet.to_csv(os.path.join(OUT_DIR, f"axial_coding_worksheet{file_suffix}.csv"),
                      index=False, encoding="utf-8-sig")
    df[["id", "tieu_de", "chu_de_bac2", "khia_canh_bac3"]].to_csv(
        os.path.join(OUT_DIR, f"axial_doc_assignment{file_suffix}.csv"),
        index=False, encoding="utf-8-sig")
    print(f"    Đã lưu: axial_coding_worksheet{file_suffix}.csv, "
          f"axial_doc_assignment{file_suffix}.csv")

def plot_mds(dist, dominant, topic_names, title, out_name):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              n_init=4, normalized_stress="auto")
    coords = mds.fit_transform(dist)
    K = len(topic_names)
    cmap = plt.cm.tab10 if K <= 10 else plt.cm.tab20
    colors = {topic_names[k]: cmap(k / max(K - 1, 1)) for k in range(K)}
    plt.figure(figsize=(9, 7))
    for t in colors:
        idx = (dominant == t).values
        plt.scatter(coords[idx, 0], coords[idx, 1], label=t, s=45, color=colors[t])
    outlier_idx = (~dominant.isin(topic_names)).values
    if outlier_idx.any():
        plt.scatter(coords[outlier_idx, 0], coords[outlier_idx, 1], label="Outlier",
                    s=30, color="lightgray", marker="x")
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=140)
    print(f"    Đã lưu hình MDS -> {out_name}")

def _bm25_weight(count_X, k1=1.5, b=0.75):
    """Trọng số BM25 (Okapi) áp cho toàn bộ ma trận đếm từ, dùng làm 1 biểu
    diễn tài liệu độc lập với TF-IDF (không phải để chấm điểm truy vấn 1 câu
    hỏi cụ thể — đây là cách dùng BM25 phổ biến trong phân cụm văn bản: coi
    mỗi từ trong từ vựng như 1 'truy vấn'). Khác TF-IDF thô ở 2 điểm: (1) tần
    suất từ được BÃO HÒA — từ lặp nhiều trong 1 tài liệu không tăng trọng số
    tuyến tính vô hạn; (2) CHUẨN HÓA theo độ dài tài liệu so với độ dài trung
    bình cả kho — hợp khi corpus có tài liệu dài ngắn chênh lệch nhiều (vd lẫn
    cả Điều luật ngắn và bài báo dài)."""
    X = count_X.tocsr().astype(float)
    doc_len = np.asarray(X.sum(axis=1)).ravel()
    avgdl = doc_len.mean() if doc_len.mean() > 0 else 1.0
    n_docs = X.shape[0]
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    Xc = X.tocoo()
    denom = Xc.data + k1 * (1 - b + b * (doc_len[Xc.row] / avgdl))
    bm25_data = idf[Xc.col] * (Xc.data * (k1 + 1) / denom)
    return csr_matrix((bm25_data, (Xc.row, Xc.col)), shape=X.shape)

# =============================================================================
# ENGINE = "tfidf_nmf"
# =============================================================================
if ENGINE == "tfidf_nmf":
    if len(df) < 4:
        raise SystemExit(
            f"[LỖI] Chỉ có {len(df)} tài liệu sau khi phân đoạn — quá ít để topic modeling "
            f"có ý nghĩa (cần tối thiểu ~15-20 trở lên). Kiểm tra lại SRC/MODE, hoặc gộp thêm "
            f"văn bản vào corpus."
        )
    df["clean"] = df["text"].apply(preprocess)
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2)
    X = tfidf.fit_transform(df["clean"])
    terms = tfidf.get_feature_names_out()
    print(f"[2] TF-IDF: {X.shape[0]} tài liệu x {X.shape[1]} từ/cụm từ.")

    if K_SCAN_REPORT:
        print(f"\n[2b] So sánh nhanh {len(K_SCAN_REPORT)} giá trị K (K_SCAN_REPORT) — không có 1")
        print("     chỉ số nào là 'đáp án đúng' một mình, đọc CẢ 3 cột cùng lúc:")
        print(f"{'K':>3} {'silhouette':>11} {'%chủ đề lớn nhất':>18} {'ARI(NMF,KMeans-TFIDF)':>23}")
        for k_try in K_SCAN_REPORT:
            if k_try < 2 or k_try >= len(df):
                print(f"{k_try:>3}  (bỏ qua — K phải trong khoảng 2..{len(df) - 1})")
                continue
            W_try = NMF(n_components=k_try, random_state=42, max_iter=400,
                        init="nndsvda").fit_transform(X)
            labels_try = W_try.argmax(axis=1)
            if len(set(labels_try)) < 2:
                print(f"{k_try:>3}  (bỏ qua — chỉ ra được <2 chủ đề phân biệt)")
                continue
            sil_try = silhouette_score(X, labels_try, metric="cosine")
            max_share = pd.Series(labels_try).value_counts().max() / len(df)
            km_tfidf_try = KMeans(n_clusters=k_try, random_state=42, n_init=10).fit_predict(X)
            ari_try = adjusted_rand_score(labels_try, km_tfidf_try)
            print(f"{k_try:>3} {sil_try:>11.3f} {max_share * 100:>17.1f}% {ari_try:>23.3f}")
        print("     silhouette cao + %chủ đề lớn nhất THẤP + ARI cao đều là dấu hiệu TỐT —")
        print("     nhưng bảng này CHỈ dùng ARI(NMF,KMeans-TFIDF) cho nhanh (bỏ qua BM25/LDA/LSA")
        print("     để không chạy chậm với nhiều K). Với K bạn quan tâm nhất từ bảng trên, đặt")
        print("     FORCE_K = K đó rồi chạy lại để xem đủ 5 phương pháp + từ khóa + tài liệu")
        print("     thật trước khi quyết định.\n")

    # chặn trên thận trọng cho corpus nhỏ (đảm bảo LUÔN có ít nhất 1 giá trị K
    # để thử, kể cả với corpus chỉ ~10 tài liệu); corpus lớn hơn nên quét rộng
    # hơn, vd range(5, 25) — sửa K_MAX_CAP bên dưới nếu muốn. Dùng //3 (không
    # phải //5) để dò xa hơn trước khi chặn — corpus càng đa dạng thể loại
    # (luật + báo + whitepaper trộn lẫn) càng dễ có nhiều chủ đề thật hơn số
    # nhỏ, quét hẹp sẽ khiến silhouette tăng đơn điệu tới đúng biên trên rồi
    # dừng (dấu hiệu K tối ưu thật có thể còn cao hơn) — xem cảnh báo bên dưới.
    K_MIN = 3
    K_MAX_CAP = 15  # từng để 9 -> đã bị chạm biên trên 2 lần thật (corpus 60 Điều
                     # VÀ corpus 27 bài báo đều dừng đúng ở K=8=biên) nên đây không
                     # còn là "chặn trên thận trọng cho corpus nhỏ" như comment gốc
                     # nữa mà đang là nút thắt thật -> nâng lên; //3 ở dưới vẫn tự
                     # giới hạn hợp lý theo cỡ corpus, K_MAX_CAP chỉ còn là chặn an
                     # toàn cho corpus rất lớn (tránh quét quá nhiều K, chậm)
    upper = max(K_MIN + 1, min(K_MAX_CAP, len(df) // 3 + 1))  # đảm bảo range không rỗng
    upper = min(upper, len(df))                                 # không vượt số tài liệu
    K_RANGE = range(K_MIN, upper) if upper > K_MIN else range(2, K_MIN + 1)

    if FORCE_K is not None:
        K = FORCE_K
        W_k = NMF(n_components=K, random_state=42, max_iter=400, init="nndsvda").fit_transform(X)
        best_score = silhouette_score(X, W_k.argmax(axis=1), metric="cosine") \
            if len(set(W_k.argmax(axis=1))) >= 2 else float("nan")
        print(f"[3] FORCE_K={K} — bỏ qua quét silhouette, dùng đúng K này "
              f"(silhouette tại K này = {best_score:.3f}).")
    else:
        best_k, best_score, scan_rows = None, -1, []
        for k in K_RANGE:
            W_k = NMF(n_components=k, random_state=42, max_iter=400, init="nndsvda").fit_transform(X)
            labels_k = W_k.argmax(axis=1)
            if len(set(labels_k)) < 2:
                continue
            score = silhouette_score(X, labels_k, metric="cosine")
            scan_rows.append({"K": k, "silhouette": round(score, 4)})
            if score > best_score:
                best_k, best_score = k, score
        if best_k is None:
            raise SystemExit(
                f"[LỖI] Không tìm được K hợp lệ trong khoảng {list(K_RANGE)} — mọi giá trị K thử "
                f"đều cho ra <2 chủ đề phân biệt (corpus có thể quá nhỏ hoặc nội dung quá đồng "
                f"nhất giữa các tài liệu). Thử giảm min_df trong TfidfVectorizer ở trên, hoặc thêm "
                f"tài liệu vào corpus."
            )
        K = best_k
        print("[3] Quét K tự động (silhouette score):")
        print(pd.DataFrame(scan_rows).to_string(index=False))
        print(f"    -> chọn K = {K} (silhouette = {best_score:.3f})")
        if K == max(K_RANGE):
            print(f"    [!] K={K} là biên TRÊN của khoảng đã quét {list(K_RANGE)}. Nếu bảng trên cho\n"
                  f"        thấy silhouette tăng ĐỀU qua nhiều K liên tiếp (không có đỉnh rõ), đó\n"
                  f"        thường KHÔNG phải dấu hiệu 'cần quét xa hơn nữa' mà là dấu hiệu\n"
                  f"        silhouette đang tiến dần về lời giải suy biến (K gần bằng số tài liệu,\n"
                  f"        mỗi cụm chỉ 1-2 tài liệu gần giống hệt nhau) — không còn ý nghĩa chọn K\n"
                  f"        khách quan nữa. Cách xử lý: đọc bảng silhouette trên, tự chọn 1 K mà bạn\n"
                  f"        thấy hợp lý về số lượng 'chủ đề bậc 2' cho mã hóa trục, rồi đặt\n"
                  f"        FORCE_K = <K đó> ở đầu file và chạy lại (bỏ qua quét tự động).")

    nmf = NMF(n_components=K, random_state=42, max_iter=500, init="nndsvda")
    W = nmf.fit_transform(X)     # N tài liệu x K chủ đề (trọng số)
    H = nmf.components_          # K chủ đề x từ vựng
    topic_names = [f"NMF-{k+1}" for k in range(K)]
    dominant = pd.Series(W.argmax(axis=1), index=df.index).map(lambda k: topic_names[k])
    df["chu_de_bac2"] = dominant

    topic_rows, topic_keywords = [], {}
    print(f"\n[4] {K} chủ đề tự phát hiện:")
    for k in range(K):
        top_idx = H[k].argsort()[::-1][:10]
        top_words = [terms[i] for i in top_idx]
        topic_keywords[topic_names[k]] = top_words
        top_doc_idx = W[:, k].argsort()[::-1][:4]
        top_docs = df.iloc[top_doc_idx]["id"].tolist()
        n_docs = int((dominant == topic_names[k]).sum())
        print(f"    {topic_names[k]} ({n_docs} tài liệu): {', '.join(top_words)}")
        print(f"        đại diện: {', '.join(top_docs)}")
        topic_rows.append({
            "chu_de": topic_names[k], "so_tai_lieu": n_docs,
            "tu_khoa": ", ".join(top_words), "tai_lieu_dai_dien": ", ".join(top_docs),
            "nhan_de_xuat": "",  # <- điền tay sau khi đọc từ khóa + tài liệu đại diện
        })
    pd.DataFrame(topic_rows).to_csv(os.path.join(OUT_DIR, "topics.csv"),
                                     index=False, encoding="utf-8-sig")
    df[["id", "tieu_de", "chu_de_bac2"]].to_csv(
        os.path.join(OUT_DIR, "doc_assignment.csv"), index=False, encoding="utf-8-sig")
    print("\n    Đã lưu: topics.csv (cột 'nhan_de_xuat' để bạn điền tay), doc_assignment.csv")

    # mạng tương quan giữa các chủ đề (Pearson trên cột trọng số W)
    W_df = pd.DataFrame(W, columns=topic_names)
    corr = W_df.corr(method="pearson")
    edges = []
    for i, a in enumerate(topic_names):
        for b in topic_names[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.34:
                edges.append({"chu_de_A": a, "chu_de_B": b, "pearson_r": round(r, 3)})
    edges_df = pd.DataFrame(edges, columns=["chu_de_A", "chu_de_B", "pearson_r"]).sort_values(
        "pearson_r", ascending=False)
    edges_df.to_csv(os.path.join(OUT_DIR, "correlation_network_edges.csv"),
                     index=False, encoding="utf-8-sig")
    print(f"\n[5] Mạng tương quan giữa chủ đề: {len(edges_df)} cặp |r| ≥ 0.34.")

    # ĐỘ VỮNG — nhiều biểu diễn/thuật toán độc lập, không có nhãn tay để so.
    # Ý tưởng: mỗi phương pháp dưới đây có GIẢ ĐỊNH khác nhau về "chủ đề là
    # gì" (phân rã ma trận tuyến tính / khoảng cách hình học / trọng số từ
    # bão hòa+chuẩn hóa độ dài / mô hình xác suất sinh) — nếu nhiều lăng kính
    # độc lập vẫn cho cùng 1 cách nhóm tài liệu, đó là bằng chứng cấu trúc
    # chủ đề là THẬT trong dữ liệu, không phải sản phẩm phụ của riêng 1 thuật
    # toán. Đây là "tam giác hóa phương pháp" quen thuộc trong nghiên cứu
    # định tính, KHÔNG phải kiểu ensemble học máy có nhãn để "vote ra đáp án
    # đúng" (ở đây không có đáp án đúng để so) — không có phương pháp nào
    # "cao cấp hơn" tuyệt đối, và ARI thấp giữa 2 phương pháp là TÍN HIỆU HỮU
    # ÍCH (cấu trúc không ổn định), không phải lỗi cần sửa cho khớp nhau.
    labels_nmf = W.argmax(axis=1)
    reps = {"NMF": labels_nmf}

    km_tfidf = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X)
    reps["KMeans-TFIDF"] = km_tfidf

    svd = TruncatedSVD(n_components=min(20, X.shape[1] - 1), random_state=42)
    km_lsa = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(svd.fit_transform(X))
    reps["KMeans-LSA"] = km_lsa

    # BM25: cùng họ với TF-IDF (thưởng từ hiếm, phạt từ phổ biến) nhưng thêm
    # bão hòa tần suất + chuẩn hóa độ dài tài liệu (xem _bm25_weight ở trên).
    # Không cần cài thêm gì — tự tính bằng công thức Okapi BM25 chuẩn.
    count_vec = CountVectorizer(vocabulary=tfidf.vocabulary_)
    X_counts = count_vec.fit_transform(df["clean"])
    X_bm25 = _bm25_weight(X_counts)
    # TfidfVectorizer L2-chuẩn hóa vector mỗi tài liệu MẶC ĐỊNH (norm="l2"),
    # nên KMeans (khoảng cách Euclid) trên TF-IDF thực chất tương đương so
    # sánh theo góc (cosine). Chuẩn hóa BM25 tương tự để KMeans-BM25 so sánh
    # trên cùng "luật chơi hình học" với KMeans-TFIDF/KMeans-LSA — nếu không,
    # chênh lệch độ dài tài liệu có thể ảnh hưởng đến khoảng cách Euclid theo
    # cách không liên quan gì đến nội dung, làm sai lệch phép so sánh.
    X_bm25 = normalize(X_bm25, norm="l2")
    km_bm25 = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X_bm25)
    reps["KMeans-BM25"] = km_bm25

    # LDA (gensim, tùy chọn): mô hình XÁC SUẤT SINH — khác hẳn giả định với
    # NMF/KMeans (đều là phân rã ma trận/khoảng cách hình học), nên là 1 lăng
    # kính độc lập thật sự, không phải biến thể của cùng 1 ý tưởng như BM25
    # với TF-IDF. Cũng là phương pháp topic modeling quen thuộc hơn với giới
    # nghiên cứu luật/khoa học xã hội — nếu bài viết cần trích phương pháp
    # luận, LDA thường dễ được người đọc/phản biện chấp nhận hơn NMF/BERTopic.
    # Dùng ĐÚNG K đã chọn ở trên (không tự quét K riêng bằng coherence) để so
    # sánh công bằng 1-đổi-1 với các phương pháp còn lại. CHƯA test được
    # trong sandbox của tôi — không cài được gensim (PyPI bị chặn) — nhưng
    # không bắt buộc: nếu bạn không cài gensim, script tự bỏ qua, 4 biểu diễn
    # còn lại vẫn đủ để đánh giá độ vững.
    lda_model, lda_available = None, False
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        lda_texts = [d.split() for d in df["clean"]]
        lda_dict = corpora.Dictionary(lda_texts)
        lda_corpus = [lda_dict.doc2bow(t) for t in lda_texts]
        lda_model = LdaModel(corpus=lda_corpus, id2word=lda_dict, num_topics=K,
                              random_state=42, passes=20)
        # get_document_topics() có thể trả về rỗng nếu 1 tài liệu không còn từ
        # nào sau preprocess() (hiếm, nhưng có thể xảy ra với tài liệu rất
        # ngắn) -> dùng -1 làm nhãn dự phòng thay vì để crash ở max() trên
        # iterable rỗng.
        lda_labels = []
        for d in lda_corpus:
            dt = lda_model.get_document_topics(d)
            lda_labels.append(max(dt, key=lambda x: x[1])[0] if dt else -1)
        reps["LDA"] = np.array(lda_labels)
        lda_available = True
    except ImportError:
        print("    [i] Bỏ qua LDA (chưa cài gensim: pip install gensim). Không bắt buộc — "
              "các biểu diễn còn lại vẫn đủ để đánh giá độ vững.")

    print(f"[6] Độ vững — {len(reps)} biểu diễn/thuật toán độc lập, không có nhãn tay:")
    names = list(reps.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            label = f"ARI({a}, {b})"
            print(f"    {label:<32}= {adjusted_rand_score(reps[a], reps[b]):.3f}")

    if lda_available:
        print("\n    Từ khóa LDA (đối chứng độc lập với từ khóa NMF ở mục [4] trên — nếu 2 bộ")
        print("    từ khóa của cùng 1 chủ đề khá giống nhau, đó là bằng chứng chủ đề đó không")
        print("    phải artefact riêng của NMF):")
        for k in range(K):
            top_lda = [w for w, _ in lda_model.show_topic(k, topn=8)]
            print(f"      LDA-{k+1}: {', '.join(top_lda)}")
        # Xuất CSV để tự xem tài liệu nào rơi vào chủ đề nào theo LDA, đối
        # chiếu trực tiếp với cột NMF trên CÙNG 1 dòng — hữu ích khi ARI(NMF,
        # LDA) thấp: đọc bảng này để biết CỤ THỂ 2 phương pháp bất đồng ở
        # những tài liệu nào, thay vì chỉ nhìn 1 con số tổng.
        pd.DataFrame({
            "id": df["id"], "tieu_de": df["tieu_de"],
            "chu_de_NMF": df["chu_de_bac2"],
            "chu_de_LDA": [f"LDA-{i+1}" for i in reps["LDA"]],
        }).to_csv(os.path.join(OUT_DIR, "doc_assignment_lda_vs_nmf.csv"),
                  index=False, encoding="utf-8-sig")
        print("    Đã lưu: doc_assignment_lda_vs_nmf.csv (so NMF và LDA theo từng tài liệu)")

    # MDS trên hình học TF-IDF (đúng không gian mà NMF thực sự dùng)
    dist = np.clip(1 - cosine_similarity(X), 0, None)
    plot_mds(dist, dominant, topic_names,
             f"MDS — hình học nội dung ({len(df)} tài liệu)\n(màu = chủ đề NMF, K={K})",
             "mds.png")

    # mã hóa trục & chọn lọc — dùng H (K x từ vựng) làm topic_vectors,
    # trọng số NMF cao nhất mỗi tài liệu làm doc_confidence
    doc_confidence = pd.Series(W.max(axis=1), index=df.index)
    aggregate_and_export(df, topic_names, H, topic_keywords, doc_confidence, file_suffix="")

# =============================================================================
# ENGINE = "bertopic"  (embedding thật + BERTopic/HDBSCAN — chưa test trong
# sandbox, xem cảnh báo ở đầu file)
# =============================================================================
elif ENGINE == "bertopic":
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as e:
        raise SystemExit(
            f"Thiếu thư viện cho ENGINE='bertopic' ({e}).\n"
            f"Cài: pip install sentence-transformers bertopic umap-learn hdbscan torch\n"
            f"Hoặc đổi ENGINE = \"tfidf_nmf\" ở đầu file để chạy ngay không cần cài thêm."
        )

    def _is_oom_error(exc) -> bool:
        return "out of memory" in str(exc).lower()

    def _encode_with_fallback(model_name, texts, device_pref, batch_size):
        """Mã hóa embedding với tự phục hồi khi GPU hết bộ nhớ (CUDA OOM):
        thử GPU (fp16) với batch_size đã cho -> nếu OOM, giảm batch_size xuống
        một nửa và thử lại trên GPU -> nếu batch_size=1 vẫn OOM, bỏ hẳn GPU và
        chạy toàn bộ trên CPU (fp32). Không có kiểu 'nửa GPU nửa CPU trong
        cùng 1 batch' vì sentence-transformers không hỗ trợ việc đó an toàn.
        """
        use_cuda = device_pref == "cuda" and torch.cuda.is_available()
        if device_pref == "cuda" and not torch.cuda.is_available():
            print("    [!] EMBED_DEVICE='cuda' nhưng không tìm thấy GPU CUDA -> chạy CPU.")
        device = "cuda" if use_cuda else "cpu"

        model = SentenceTransformer(model_name, device=device)
        bs = batch_size
        while True:
            try:
                if device == "cuda":
                    model = model.half()  # fp16: giảm ~2 lần bộ nhớ GPU
                    print(f"    Mã hóa trên GPU (fp16), batch_size={bs}...")
                else:
                    print(f"    Mã hóa trên CPU, batch_size={bs}...")
                return model.encode(texts, show_progress_bar=True, batch_size=bs,
                                     convert_to_numpy=True)
            except RuntimeError as e:
                if not _is_oom_error(e) or device != "cuda":
                    raise
                torch.cuda.empty_cache()
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"    [!] GPU hết bộ nhớ (CUDA OOM) -> thử lại với batch_size={bs}...")
                    model = model.float()  # reset trước khi .half() lại ở vòng sau
                    continue
                print("    [!] GPU vẫn hết bộ nhớ ngay cả với batch_size=1 -> "
                      "chuyển toàn bộ sang CPU (chậm hơn nhưng chắc chắn chạy được).")
                del model
                torch.cuda.empty_cache()
                device = "cpu"
                bs = batch_size
                model = SentenceTransformer(model_name, device="cpu")

    df["seg"] = df["text"].apply(segment_for_embedding)
    print(f"[2] Tách từ cho embedding xong. Đang tải model '{BERTOPIC_MODEL_NAME}' "
          f"(lần đầu cần mạng, ~500MB-2GB tùy model)...")
    device_pref = "cpu" if EMBED_DEVICE == "cpu" else ("cuda" if EMBED_DEVICE in ("cuda", "auto") else "cpu")
    embeddings = _encode_with_fallback(BERTOPIC_MODEL_NAME, df["seg"].tolist(),
                                        device_pref, EMBED_BATCH_SIZE)
    print(f"[3] Đã tạo {embeddings.shape[0]} embedding, chiều {embeddings.shape[1]}.")

    min_topic_size = BERTOPIC_MIN_TOPIC_SIZE or max(3, len(df) // 15)
    n_neighbors = min(15, max(2, len(df) - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, metric="cosine",
                       random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                             cluster_selection_method="eom", prediction_data=True)
    # lọc stopword khi trích từ khóa c-TF-IDF (không ảnh hưởng embedding, chỉ
    # ảnh hưởng phần hiển thị "chủ đề gồm những từ gì")
    cv_stopwords = list(STOPWORDS) + [w.replace(" ", "_") for w in STOPWORDS if " " in w]
    vectorizer_model = CountVectorizer(stop_words=cv_stopwords, token_pattern=r"(?u)\b\w\w+\b")

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                            vectorizer_model=vectorizer_model,
                            calculate_probabilities=False, verbose=True)
    topics, _ = topic_model.fit_transform(df["seg"].tolist(), embeddings=embeddings)
    df["_topic_id"] = topics

    valid_ids = sorted(t for t in set(topics) if t != -1)
    K = len(valid_ids)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"[4] HDBSCAN tự phát hiện {K} chủ đề (không khai báo K), "
          f"{n_outliers}/{len(df)} tài liệu là 'nhiễu' (outlier, không thuộc chủ đề nào).")
    if len(df) and n_outliers / len(df) > 0.5:
        print("    CẢNH BÁO: hơn 50% tài liệu là outlier — thử giảm "
              "BERTOPIC_MIN_TOPIC_SIZE ở đầu file.")

    id_to_name = {t: f"BT-{t}" for t in valid_ids}
    topic_names = [id_to_name[t] for t in valid_ids]
    dominant = pd.Series([id_to_name.get(t, "Outlier") for t in topics], index=df.index)
    df["chu_de_bac2"] = dominant

    # Từ khóa đại diện: KHÔNG dùng topic_model.get_topic() — trên thực tế đã
    # thấy get_topic() trả về từ tiếng Việt bị mất dấu VÀ mất một số nguyên âm
    # (vd "công nghệ" -> "cngngh"), lặp lại nhất quán giữa các chủ đề, không
    # phải lỗi hiển thị console (thấy cả trong topics_bertopic.csv) — nhiều khả
    # năng là tương tác giữa CountVectorizer tùy biến và bước c-TF-IDF nội bộ
    # của BERTopic trên phiên bản bạn cài, nhưng tôi không có môi trường
    # sentence-transformers/bertopic để debug trực tiếp (sandbox không cài
    # được các thư viện này) nên không khẳng định chắc nguyên nhân gốc. Thay
    # vì cố sửa bên trong BERTopic, tự tính c-TF-IDF THỦ CÔNG bằng đúng cơ chế
    # TfidfVectorizer đã kiểm chứng cho ra tiếng Việt đúng ở ENGINE="tfidf_nmf":
    # coi mỗi chủ đề là 1 "văn bản gộp" (nối các tài liệu thuộc chủ đề đó), rồi
    # TF-IDF giữa các văn bản gộp này để tìm từ đặc trưng riêng từng chủ đề.
    df["_clean_kw"] = df["text"].apply(preprocess)
    topic_docs = [" ".join(df.loc[df["_topic_id"] == t, "_clean_kw"]) for t in valid_ids]
    kw_vectorizer = TfidfVectorizer(min_df=1, stop_words=list(STOPWORDS) +
                                     [w.replace(" ", "_") for w in STOPWORDS if " " in w])
    kw_X = kw_vectorizer.fit_transform(topic_docs)
    kw_terms = kw_vectorizer.get_feature_names_out()

    topic_keywords, topic_rows = {}, []
    for row_i, t in enumerate(valid_ids):
        top_idx = kw_X[row_i].toarray().ravel().argsort()[::-1][:10]
        words = [kw_terms[i] for i in top_idx]
        topic_keywords[id_to_name[t]] = words
        idxs = [i for i, tt in enumerate(topics) if tt == t]
        centroid = embeddings[idxs].mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeddings[idxs], centroid).ravel()
        top4 = [idxs[j] for j in np.argsort(-sims)[:4]]
        top_docs = df.iloc[top4]["id"].tolist()
        print(f"    {id_to_name[t]} ({len(idxs)} tài liệu): {', '.join(words)}")
        print(f"        đại diện: {', '.join(top_docs)}")
        topic_rows.append({
            "chu_de": id_to_name[t], "so_tai_lieu": len(idxs),
            "tu_khoa": ", ".join(words), "tai_lieu_dai_dien": ", ".join(top_docs),
            "nhan_de_xuat": "",
        })
    pd.DataFrame(topic_rows).to_csv(os.path.join(OUT_DIR, "topics_bertopic.csv"),
                                     index=False, encoding="utf-8-sig")
    df[["id", "tieu_de", "chu_de_bac2"]].to_csv(
        os.path.join(OUT_DIR, "doc_assignment_bertopic.csv"), index=False, encoding="utf-8-sig")
    print("\n    Đã lưu: topics_bertopic.csv, doc_assignment_bertopic.csv")

    # độ vững: HDBSCAN(BERTopic) vs KMeans trên cùng embedding vs KMeans trên
    # TF-IDF của cùng văn bản đã tách từ — chỉ so trên tập tài liệu KHÔNG bị
    # coi là outlier (KMeans không có khái niệm outlier nên không so được)
    print(f"\n[5] Độ vững — so trên {len(df) - n_outliers} tài liệu không phải outlier:")
    if K >= 2:
        mask = np.array(topics) != -1
        km_embed = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(embeddings)
        tfidf_bt = TfidfVectorizer(max_df=0.9, min_df=2)
        X_bt = tfidf_bt.fit_transform(df["text"].apply(preprocess))
        km_tfidf_bt = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X_bt)
        labels_bt = np.array(topics)
        print(f"    ARI(HDBSCAN, KMeans-embedding) = "
              f"{adjusted_rand_score(labels_bt[mask], km_embed[mask]):.3f}")
        print(f"    ARI(HDBSCAN, KMeans-TFIDF)     = "
              f"{adjusted_rand_score(labels_bt[mask], km_tfidf_bt[mask]):.3f}")
    else:
        print("    Bỏ qua (K < 2 chủ đề hợp lệ — thử giảm BERTOPIC_MIN_TOPIC_SIZE).")

    # MDS trên hình học embedding thật (PhoBERT)
    dist = np.clip(1 - cosine_similarity(embeddings), 0, None)
    plot_mds(dist, dominant, topic_names,
             f"MDS — hình học embedding ({BERTOPIC_MODEL_NAME}, {len(df)} tài liệu)\n"
             f"(màu = chủ đề BERTopic/HDBSCAN, K={K} tự phát hiện)",
             "mds_bertopic.png")

    # mã hóa trục & chọn lọc — dùng tâm cụm embedding mỗi chủ đề làm
    # topic_vectors; độ tương đồng cosine của tài liệu với tâm cụm của
    # chính chủ đề nó làm doc_confidence
    topic_vectors = np.array([
        embeddings[[i for i, tt in enumerate(topics) if tt == t]].mean(axis=0)
        for t in valid_ids
    ])
    conf = np.zeros(len(df))
    for t in valid_ids:
        idxs = [i for i, tt in enumerate(topics) if tt == t]
        centroid = embeddings[idxs].mean(axis=0, keepdims=True)
        conf[idxs] = cosine_similarity(embeddings[idxs], centroid).ravel()
    doc_confidence = pd.Series(conf, index=df.index)

    if K >= 2:
        aggregate_and_export(df, topic_names, topic_vectors, topic_keywords,
                              doc_confidence, file_suffix="_bertopic")
    else:
        print("\n    Bỏ qua bước mã hóa trục (cần ít nhất 2 chủ đề hợp lệ).")

else:
    raise ValueError("ENGINE phải là 'tfidf_nmf' hoặc 'bertopic'")

# =============================================================================
# HƯỚNG DẪN BƯỚC TIẾP THEO — KHÔNG THỂ CODE HÓA HỢP LÝ (phần của bạn)
# =============================================================================
print("\n" + "=" * 70)
print("BƯỚC TIẾP THEO — KHÔNG THỂ CODE HÓA HỢP LÝ (đây là phần của bạn):")
print("  1) Mở topics*.csv: đọc 'tu_khoa' + 'tai_lieu_dai_dien', điền 'nhan_de_xuat'")
print("     cho từng chủ đề bậc 2.")
print("  2) Mở axial_coding_worksheet*.csv: với mỗi khía cạnh bậc 3, đọc 'tu_khoa_gop'")
print("     + 'bang_chung_dai_dien', điền 'ten_de_xuat', rồi quyết định 'vai_tro_paradigm'")
print("     (khía cạnh tỷ trọng bất thường cao/thấp thường là ứng viên tốt cho 'điều")
print("     kiện nhân quả' hoặc 'hiện tượng cốt lõi' — nhưng cần bạn đọc bằng chứng và")
print("     quyết định, đó là lý luận khoa học, không phải phép tính).")
print("  3) 'Trục' (vd: thủ tục hành chính <-> nội dung hợp đồng) và 'hiện trạng' cho")
print("     bài viết là kết luận rút ra từ worksheet đã điền đầy đủ ở bước 2 — không")
print("     phải output trực tiếp của script.")
print("=" * 70)