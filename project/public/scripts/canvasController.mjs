/**
 * @typedef {import("./common.mjs").Coordinates} Coordinates
 */

/**
 * @typedef CanvasController
 * @type {object}
 * @property {() => void} setup
 * @property {(coordinates: Coordinates) => Promise<void>} update
 */

/**
 * @typedef Params
 * @type {object}
 * @property {HTMLCanvasElement} canvas
 * @property {number} errorRadius
 * @property {number} targetRadius
 */

/**
 * @typedef Self
 * @type {object}
 * @property {HTMLCanvasElement} canvas
 * @property {CanvasRenderingContext2D} context
 * @property {number} errorRadius
 * @property {number} targetRadius
 * @property {number} width
 * @property {number} height
 * @property {number} dpi
 * @property {() => Promise<void>} onResize
 */

/**
 * @param {Params} params
 * @return {CanvasController}
 */
export const makeCanvasController = (params) => {
    /**
     * @type {Self}
     */
    const self = {
        canvas: params.canvas,
        context: params.canvas.getContext('2d'),
        errorRadius: params.errorRadius,
        targetRadius: params.targetRadius,
        width: params.width,
        height: params.height,
        dpi: params.dpi,
        onResize: null
    }
    self.onResize = onResize(self)

    return {
        setup: setup(self),
        update: update(self)
    }
}

/**
 * @param {Self} self
 * @return {Self['onResize']}
 */
const onResize = (self) => () => {
    self.dpi = window.devicePixelRatio
    self.width = document.body.clientWidth * self.dpi
    self.height = document.body.clientHeight * self.dpi

    self.canvas.width = self.width
    self.canvas.height = self.height

    self.context.fillStyle = '#000000'
    self.context.fillRect(0, 0, self.width, self.height)
    self.context.scale(self.dpi, self.dpi)
}

/**
 * @param {Self} self
 * @return {CanvasController['setup']}
 */
const setup = (self) => () => {
    window.addEventListener('resize', self.onResize)
    self.onResize()
}

/**
 * @param {Self} self
 * @return {CanvasController['update']}
 */
const update = (self) => async coordinates => {
    const x = coordinates.x
    const y = coordinates.y

    console.log({ x, y }, coordinates.x, coordinates.y)

    self.context.fillStyle = '#000000'
    self.context.fillRect(0, 0, self.width, self.height)

    self.context.strokeStyle = '#C0C0C0'
    self.context.beginPath()
    self.context.arc(x, y, self.errorRadius, 0, 2 * Math.PI)
    self.context.stroke()

    self.context.fillStyle = '#FFFFFF'
    self.context.beginPath()
    self.context.arc(x, y, self.targetRadius, 0, 2 * Math.PI)
    self.context.fill()
}
