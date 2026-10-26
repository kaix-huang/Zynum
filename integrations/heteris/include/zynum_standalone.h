/* SPDX-License-Identifier: LGPL-3.0-or-later */
#ifndef ZYNUM_STANDALONE_H
#define ZYNUM_STANDALONE_H
#include "zynum.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Optional standalone SDK. Link zynum-standalone explicitly; libzynum contains
 * only mathematical operations and never creates this provider implicitly. */
/* Explicit lightweight malloc/free provider, no pools. A destroy with live
 * allocations returns BUSY and leaves the resource table valid. */
heteris_status zynum_resources_create_naive(zynum_resources *out);
heteris_status zynum_resources_destroy_naive(zynum_resources *resources);

#ifdef __cplusplus
}
#endif
#endif
