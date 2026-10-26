#include "ecg_proto.h"

#include <string.h>

enum { S_SYNC0, S_SYNC1, S_HEADER, S_PAYLOAD, S_CRC };

uint16_t proto_crc16(uint16_t crc, const uint8_t *data, size_t n)
{
    while (n--) {
        crc ^= (uint16_t)(*data++ << 8);
        for (int i = 0; i < 8; i++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

void proto_parser_init(proto_parser_t *p)
{
    memset(p, 0, sizeof(*p) - sizeof(p->frame));
    p->state = S_SYNC0;
}

int proto_feed(proto_parser_t *p, uint8_t b)
{
    proto_frame_t *f = &p->frame;
    switch (p->state) {
    case S_SYNC0:
        if (b == PROTO_SYNC0)
            p->state = S_SYNC1;
        return 0;
    case S_SYNC1:
        p->state = (b == PROTO_SYNC1) ? S_HEADER : (b == PROTO_SYNC0 ? S_SYNC1 : S_SYNC0);
        p->got = 0;
        return 0;
    case S_HEADER:
        /* type, seq, len lo, len hi - kept in the frame struct as they arrive */
        if (p->got == 0) f->type = b;
        else if (p->got == 1) f->seq = b;
        else if (p->got == 2) f->len = b;
        else f->len |= (uint16_t)(b << 8);
        if (++p->got < 4)
            return 0;
        if (f->len > PROTO_MAX_PAYLOAD) {
            p->crc_errors++;
            p->state = S_SYNC0;
            return 0;
        }
        p->got = 0;
        p->state = f->len ? S_PAYLOAD : S_CRC;
        return 0;
    case S_PAYLOAD:
        f->payload[p->got++] = b;
        if (p->got == f->len) {
            p->got = 0;
            p->state = S_CRC;
        }
        return 0;
    default: /* S_CRC */
        if (p->got++ == 0) {
            p->crc_rx = b;
            return 0;
        }
        p->crc_rx |= (uint16_t)(b << 8);
        p->state = S_SYNC0;
        {
            uint8_t hdr[4] = {f->type, f->seq, (uint8_t)f->len, (uint8_t)(f->len >> 8)};
            uint16_t crc = proto_crc16(0xFFFF, hdr, 4);
            crc = proto_crc16(crc, f->payload, f->len);
            if (crc != p->crc_rx) {
                p->crc_errors++;
                return 0;
            }
        }
        return 1;
    }
}

size_t proto_encode(uint8_t *out, uint8_t type, uint8_t seq, const uint8_t *payload, uint16_t len)
{
    out[0] = PROTO_SYNC0;
    out[1] = PROTO_SYNC1;
    out[2] = type;
    out[3] = seq;
    wr_u16(out + 4, len);
    if (len && out + 6 != payload)
        memmove(out + 6, payload, len);
    uint16_t crc = proto_crc16(0xFFFF, out + 2, 4u + len);
    wr_u16(out + 6 + len, crc);
    return (size_t)len + PROTO_OVERHEAD;
}
