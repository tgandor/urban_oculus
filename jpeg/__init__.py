import os
from . import markers
from .quantization.ijg_tables import QT1_to_Q, QT2_to_Q, Q_to_QT1, Q_to_QT2  # noqa.

"""
References:
https://stackoverflow.com/questions/1557071/determining-the-size-of-a-jpeg-jfif-image#1602428
https://www.impulseadventure.com/photo/jpeg-decoder.html
"""


def get_QTs(filename, with_id=False):
    results = []
    offset = 1 - int(with_id)
    with open(filename, "rb") as f:
        c = ord(f.read(1))
        d = ord(f.read(1))
        assert c == 0xFF, f"{filename}: JPG needs to start with SOI"
        assert d == markers.markers["SOI"], f"{filename}: JPG needs to start with SOI"
        while True:
            c = ord(f.read(1))
            if c == 0xFF:
                d = ord(f.read(1))
                if d in markers.standalone:
                    continue
                hi = ord(f.read(1))
                lo = ord(f.read(1))
                size = 256 * hi + lo
                data = f.read(size - 2)  # 2 bytes already there
                name = markers.names.get(d, hex(d))
                if name == "DQT":
                    if size == 67:  # single QT
                        results.append(data[offset:].hex())
                    elif size == 132:  # double QT
                        results.append(data[offset:65].hex())
                        results.append(data[offset + 65 :].hex())
                    else:
                        raise ValueError("Malformed DQT")
                if name == "SOS":
                    break
    return results


def _zigzag(n):
    """Produce zig-zag coordinates (i, j) for (n x n) table."""
    nzigs = 2 * n - 1
    for zig in range(nzigs):
        if zig % 2 == 0:
            # odd (zigs): up (-1), right (1)
            di, dj = -1, 1
            i = min(zig, n - 1)
            j = zig - i
        else:
            # even (zags): down (1), left (-1)
            di, dj = 1, -1
            j = min(zig, n - 1)
            i = zig - j

        # length: zig + 1 - growing, nzigs - zig - shrinking
        len_zig = min(zig + 1, nzigs - zig)

        # print('zig', zig)
        for _ in range(len_zig):
            yield i, j
            i, j = i + di, j + dj


def reorder(hex_src):
    gen = _zigzag(8)
    dest = [[0 for i in range(8)] for j in range(8)]

    for (ti, tj), value in zip(gen, bytes.fromhex(hex_src)):
        dest[ti][tj] = value

    return dest


def get_QTs2d(filename):
    return [reorder(qt) for qt in get_QTs(filename)]


def identify_quality(filename):
    return int(os.popen(f'identify -format %Q "{filename}"').read())


def recognize_QT_quality(filename, failsafe=False):
    try:
        qts = get_QTs(filename)
    except Exception as e:
        if failsafe:
            print(filename, e)
            return 0
        raise

    if not qts:
        return -1 if failsafe else None

    q = QT1_to_Q.get(qts[0])

    if q is None:
        return 0 if failsafe else None

    if len(qts) >= 1 and qts[1] != Q_to_QT2[q]:
        print(f"{filename}: QT2 should match for Q={q}")
        print(f"    QT1={qts[0]}\n    QT2={qts[1]}\n and not {Q_to_QT2[q]}")
        if not failsafe:
            raise ValueError(f"{filename}: QT2 should match for Q={q}")

    return q


def opencv_degrade(orig, filename, q, grayscale=False):
    import cv2

    img = cv2.imread(
        orig,
        cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED,
    )
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def mogrify_degrade(orig, filename, q, grayscale=False):
    assert orig.endswith(".jpg"), "mogrify_degrade() only works with JPG files."
    import shutil

    gray = "-type Grayscale" if grayscale else ""
    shutil.copy(orig, filename)
    os.system(f"mogrify {gray} -quality {q} {filename}")


def opencv_degrade_image(image, quality):
    import cv2

    data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image
