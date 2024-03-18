/*
 * Business logic script
 *
 * Makes the page fullscreen, tracks the cursor and sends the cursor position on click.
 * The `state.scale` property in `main` needs to be adjusted.
 * 
 */

const main = async () => {
    const body = document.body
    const state = {
        scale: 1.5, // window.devicePixelRatio
        captured: 0,
        element: document.querySelector("#screen-log"),
        refresh: null
    }
    state.refresh = refresh(state)

    body.addEventListener("mousemove", mouseMoveListener(state))
    body.addEventListener("mouseup", mouseUpListener(state))
}

const refresh = state => (x, y) => {
    state.element.innerText = `
        Unscaled screen X/Y: ${x}, ${y}
        Scaled screen X/Y: ${Math.round(x * state.scale)}, ${Math.round(y * state.scale)}
        Captured: ${state.captured}
    `;
}

const mouseMoveListener = (state) => (event) => state.refresh(event.screenX, event.screenY)

const mouseUpListener = (state) => async (event) => {
    const strategy = strategyPicker()
    await strategy(state, event)
}

const strategyPicker = () => {
    if (!document.fullscreenElement) {
        return goFullscreen
    }

    return send
}

const goFullscreen = async () => {
    return await document.body.requestFullscreen({ navigationUI: "hide" })
}

const send = async (state, event) => {
    const x = Math.round(event.screenX * state.scale)
    const y = Math.round(event.screenY * state.scale)
    
    await fetch("http://localhost:3000", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ x, y })
    })
    state.captured++
    state.refresh(event.screenX, event.screenY)
}

main()