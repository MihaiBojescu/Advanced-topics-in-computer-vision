/**
 * @typedef CameraController
 * @type {object}
 * @property {() => Promise<void>} start
 * @property {() => Promise<void>} stop
 * @property {(callback: (contentType: string, data: string) => Promise<void>) => void} onFrame
 */

/**
 * @typedef Params
 * @type {object}
 * @property {HTMLVideoElement} video
 */

/**
 * @typedef Self
 * @type {object}
 * @property {HTMLVideoElement} video
 * @property {HTMLCanvasElement} canvas
 * @property {CanvasRenderingContext2D} context
 * @property {MediaStream | null} stream
 * @property {number} width
 * @property {number} height
 * @property {((contentType: string, data: string) => Promise<void>)[]} callbacks
 */

const THIRTY_FPS_TIMING_MS = 33

/**
 * @param {Params} params
 * @return {CameraController}
 */
export const makeCameraController = (params) => {
    const canvas = document.createElement('canvas')

    /**
     * @type {Self}
     */
    const self = {
        video: params.video,
        canvas: canvas,
        context: canvas.getContext('2d'),
        stream: null,
        width: -1,
        height: -1,
        callbacks: []
    }

    return {
        start: start(self),
        stop: stop(self),
        onFrame: onFrame(self)
    }
}

/**
 * @param {Self} self
 * @return {CameraController['start']}
 */
const start = self => async () => {
    self.stream = await navigator.mediaDevices.getUserMedia({ video: true })
    self.video.srcObject = self.stream
    self.video.play()

    await new Promise(resolve => {
        const canPlay = () => {
            self.width = self.video.videoWidth
            self.height = self.video.videoHeight
            self.video.removeEventListener('canplay', canPlay)

            resolve()
        }

        self.video.addEventListener('canplay', canPlay)
    })

    self.canvas.width = self.width
    self.canvas.height = self.height

    while (self.stream) {
        self.context.drawImage(self.video, 0, 0, self.width, self.height)

        const data = self.canvas.toDataURL("image/jpeg");

        for await (const callback of self.callbacks) {
            await callback("image/jpeg", data)
        }

        await new Promise(resolve => setTimeout(resolve, THIRTY_FPS_TIMING_MS))
    }
}

/**
 * @param {Self} self
 * @return {CameraController['stop']}
 */
const stop = self => async () => {
    if (!self.stream) {
        return
    }

    const tracks = self.stream.getTracks()

    for (const track of tracks) {
        track.stop()
    }

    self.video.srcObject = null
    self.stream = null
    self.width = -1
    self.height = -1
}

/**
 * @param {Self} self
 * @return {CameraController['onFrame']}
 */
const onFrame = self => callback => {
    self.callbacks.push(callback)
}
