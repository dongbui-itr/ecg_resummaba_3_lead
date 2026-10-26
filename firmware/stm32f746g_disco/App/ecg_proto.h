/*
 * PC <-> kit link over the ST-LINK virtual COM port (USART1, 8N1). Every message is one frame:
 *
 *   0xA5 0x5A | type u8 | seq u8 | len u16 | payload[len] | crc16 u16
 *
 * little-endian, crc16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over type..payload.
 * A frame with a bad CRC is dropped and the parser hunts for the next 0xA5 0x5A. The Python
 * side of the same protocol is webservice/stm32_link.py; the payloads are described there
 * and in README.md.
 *
 *   PC -> kit   HELLO  0x01  (empty)          reset the stream, answer INFO
 *               CHUNK  0x02  ECG + labels     index 0: 10 s, then 9 s each
 *   kit -> PC   INFO   0x81  model / stream parameters
 *               RESULT 0x82  one window's prediction + the labels echoed back
 *               ERROR  0x83  u8 code, u32 chunk index, text
 *               LOG    0x84  text
 */
#ifndef ECG_PROTO_H
#define ECG_PROTO_H

#include <stddef.h>
#include <stdint.h>

#define PROTO_VERSION     1
#define PROTO_SYNC0       0xA5
#define PROTO_SYNC1       0x5A
#define PROTO_OVERHEAD    8              /* sync 2 + type + seq + len 2 + crc 2 */
#define PROTO_MAX_PAYLOAD 16384

enum {
    MSG_HELLO = 0x01,
    MSG_CHUNK = 0x02,
    MSG_INFO = 0x81,
    MSG_RESULT = 0x82,
    MSG_ERROR = 0x83,
    MSG_LOG = 0x84,
};

enum {
    ERR_CHUNK_SIZE = 1,      /* first chunk not 10 s, later chunk not 9 s, or bad lead count */
    ERR_SEQUENCE = 2,        /* chunk index is not the next one (send HELLO + index 0) */
    ERR_MODEL = 3,           /* the network failed to run */
    ERR_UNKNOWN_TYPE = 4,
};

typedef struct {
    uint8_t type, seq;
    uint16_t len;
    uint8_t payload[PROTO_MAX_PAYLOAD];
} proto_frame_t;

typedef struct {
    int state;
    uint16_t got;
    uint16_t crc_rx;
    uint32_t crc_errors;
    proto_frame_t frame;
} proto_parser_t;

uint16_t proto_crc16(uint16_t crc, const uint8_t *data, size_t n);
void proto_parser_init(proto_parser_t *p);
/* One received byte; returns 1 when p->frame holds a complete frame with a good CRC. */
int proto_feed(proto_parser_t *p, uint8_t byte);
/* Frame `payload` into `out` (room for len + PROTO_OVERHEAD bytes); returns the frame length. */
size_t proto_encode(uint8_t *out, uint8_t type, uint8_t seq, const uint8_t *payload, uint16_t len);

/* Little-endian field helpers. */
static inline uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint8_t *wr_u8(uint8_t *p, uint8_t v) { *p = v; return p + 1; }
static inline uint8_t *wr_u16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); return p + 2; }
static inline uint8_t *wr_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
    return p + 4;
}

#endif
