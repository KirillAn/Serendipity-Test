let minNodeSize = 1;
let maxNodeSize = 3;
let minVelocity = 0.07; // Reduced minimum velocity
let maxVelocity = 0.11; // Reduced maximum velocity
let nodesAmount = 1000;
let backgroundColor = '#000000'; // Black background
let nodeColor = '#FFFFFF'; // White particles
let lineColor = '#FFFFFF'; // White lines

const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const nodes = [];
const createNode = () => {
    const nodeX = Math.random() * canvas.width;
    const nodeY = Math.random() * canvas.height;
    const nodeVelocityX = (Math.random() * maxVelocity + minVelocity) * (Math.random() > 0.5 ? 1 : -1);
    const nodeVelocityY = (Math.random() * maxVelocity + minVelocity) * (Math.random() > 0.5 ? 1 : -1);
    const nodeRadius = Math.random() * (maxNodeSize - minNodeSize) + minNodeSize;
    return { x: nodeX, y: nodeY, r: nodeRadius, vx: nodeVelocityX, vy: nodeVelocityY };
};

while (nodes.length < nodesAmount) {
    nodes.push(createNode());
}

const draw = () => {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
        ctx.fillStyle = nodeColor;
        ctx.fill();
        node.x += node.vx;
        node.y += node.vy;

        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
    });

    nodes.forEach((node, i) => {
        for (let j = i + 1; j < nodes.length; j++) {
            const otherNode = nodes[j];
            const distanceX = node.x - otherNode.x;
            const distanceY = node.y - otherNode.y;
            const distance = Math.sqrt(distanceX ** 2 + distanceY ** 2);
            if (distance < 150) {
                ctx.beginPath();
                ctx.moveTo(node.x, node.y);
                ctx.lineTo(otherNode.x, otherNode.y);
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 0.1;
                ctx.stroke();
            }
        }
    });

    requestAnimationFrame(draw);
};

draw();
