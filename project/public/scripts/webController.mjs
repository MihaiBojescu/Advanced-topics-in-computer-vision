/**
 * @typedef {import("./common.mjs").Coordinates} Coordinates
 */

/**
 * @typedef WebController
 * @type {object}
 * @property {(contentType: string, data: string) => Promise<Coordinates>} send
 * @property {(coordinates: Coordinates) => Promise<void>} onResponse
 */

/**
 * @typedef Params
 * @type {object}
 * @property {string} url
 */

/**
 * @typedef Self
 * @type {object}
 * @property {string} url
 * @property {((coordinates: Coordinates) => Promise<void>)[]} callbacks
 */

/**
 * @param {Params} params
 * @return {WebController}
 */
export const makeWebController = (params) => {
    /**
     * @type {Self}
     */
    const self = {
        url: params.url,
        callbacks: []
    }

    return {
        send: send(self),
        onResponse: onResponse(self)
    }
}

/**
 * @param {Self} self
 * @return {WebController['send']}
 */
const send = self => async (contentType, data) => {
    console.log({ contentType, data })
    const response = await fetch(self.url, {
        method: 'POST',
        headers: {
            'Content-type': contentType || 'application/octet-stream'
        },
        body: data
    })
    /**
     * @type {Coordinates}
     */
    const coordinates = response.json()

    for await (const callback of self.callbacks) {
        await callback(coordinates)
    }

    return coordinates
}

/**
 * @param {Self} self
 * @return {WebController['onResponse']}
 */
const onResponse = self => callback => {
    self.callbacks.push(callback)
}
