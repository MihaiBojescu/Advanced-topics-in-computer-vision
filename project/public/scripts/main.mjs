import { makeCanvasController } from "./canvasController.mjs"
import { makeCameraController } from "./cameraController.mjs"
import { makeWebController } from "./webController.mjs"

const main = async () => {
    const canvas = document.querySelector('#canvas')
    const video = document.querySelector('#video')

    if (!canvas || !('getContext' in canvas)) {
        throw new Error('Canvas element not found')
    }

    if (!video) {
        throw new Error('Video element not found')
    }

    const canvasController = makeCanvasController({
        canvas: canvas,
        errorRadius: 30,
        targetRadius: 10,
    })
    const cameraController = makeCameraController({
        video: video
    })
    const webController = makeWebController({
        url: '/predict'
    })

    cameraController.onFrame(webController.send)
    webController.onResponse(canvasController.update)

    canvasController.setup()
    await cameraController.start()
}

main()
