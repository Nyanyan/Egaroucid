var instance;
var hw = 8;
var hw2 = 64;
let dy = [0, 1, 0, -1, 1, 1, -1, -1];
let dx = [1, 0, -1, 0, 1, -1, 1, -1];
let grid = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
];
let bef_grid = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
];
var n_stones = 4;
var player = 0;
var ai_player = -1;
var level_idx = 0;
let level_names = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9', 'Level 10', 'Level 11', 'Level 12', 'Level 13', 'Level 14', 'Level 15'];
var game_end = false;
var value_calced = false;
var div_mcts = 20;
var mcts_progress = 0;
var interval_id = -1;
let record = [];
var step = 0;
var isstart = true;
var show_value = true;
var show_graph = true;
var ai_initialized = 1;
let graph_values = [];
var ctx = document.getElementById("graph");
var graph = new Chart(ctx, {
    type: 'line',
    data: {
    labels: [],
    datasets: [
        {
        label: 'AI will win/lose by ',
        data: [],
        fill: false,
        borderColor: "rgb(0,0,0)",
        backgroundColor: "rgb(0,0,0)"
        }
    ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        title: {
            display: false
        },
        legend: {
            display: false
        },
        scales: {
            yAxes: [{
            ticks: {
                suggestedMax: 10.0,
                suggestedMin: -10.0,
                stepSize: 10.0,
                callback: function(value, index, values){
                    return  value
                }
            }
            }]
        },
    }
});

const level_range = document.getElementById('ai_level');
const level_show = document.getElementById('ai_level_label');
const custom_setting = document.getElementById('custom');
const setCurrentValue = (val) => {
    level_show.innerText = level_names[val];
}

const rangeOnChange = (e) =>{
    setCurrentValue(e.target.value);
}

const setCurrentValue_book = (val) => {
    book_label.innerText = book_label.innerText = book_range.value + '手';
}

const rangeOnChange_book = (e) =>{
    setCurrentValue_book(e.target.value);
}

function start() {
    for (var y = 0; y < hw; ++y){
        for (var x = 0; x < hw; ++x) {
            grid[y][x] = -1;
            bef_grid[y][x] = -1;
        }
    }
    graph.data.labels = [];
    graph.data.datasets[0].data = [];
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    graph.data.values = [];
    graph.data.labels = [];
    graph.update();
    game_end = false;
    document.getElementById('start').disabled = true;
    level_range.disabled = true;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = true;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = true;
    show_graph = show_graph_elem.checked;
    record = [];
    document.getElementById('record').innerText = '';
    ai_player = -1;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i) {
        players.item(i).disabled = true;
        if (players.item(i).checked) {
            ai_player = players.item(i).value;
        }
    }
    console.log("ai player", ai_player);
    level_idx = level_range.value;
    console.log("level", level_idx);
    n_stones = 4;
    show(-1, -1);
    setInterval(ai_check, 250);
}

function show(r, c) {
    var table = document.getElementById("board");
    if (!check_mobility()) {
        player = 1 - player;
        if (!check_mobility()) {
            player = 2;
        }
    }
    for (var y = 0; y < 8; ++y) {
        for (var x = 0; x < 8; ++x) {
            table.rows[y].cells[x].style.backgroundColor = "#249972";
            if (grid[y][x] == 0) {
                if (bef_grid[y][x] != 0) {
                    table.rows[y].cells[x].innerHTML = '<span class="black_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 1) {
                if (bef_grid[y][x] != 1) {
                    table.rows[y].cells[x].innerHTML = '<span class="white_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 2) {
                if (r == -1 || inside(r, c)) {
                    if (player == 0) {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_black";
                    } else {
                        table.rows[y].cells[x].firstChild.className ="legal_stone_white";
                    }
                    table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
                } else {
                    //table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                    table.rows[y].cells[x].firstChild.className ="empty_stone";
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            } else {
                //table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                table.rows[y].cells[x].firstChild.className ="empty_stone";
                table.rows[y].cells[x].setAttribute('onclick', "");
            }
        }
    }
    if (inside(r, c)) {
        table.rows[r].cells[c].style.backgroundColor = "#d14141";
    }
    var black_count = 0, white_count = 0;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 0)
                ++black_count;
            else if (grid[y][x] == 1)
                ++white_count;
        }
    }
    table = document.getElementById("status");
    table.rows[0].cells[2].firstChild.innerHTML = black_count;
    table.rows[0].cells[4].firstChild.innerHTML = white_count;
    if (player == 0) {
        table.rows[0].cells[0].firstChild.className = "legal_stone";
        table.rows[0].cells[6].firstChild.className = "state_blank";
    } else if (player == 1) {
        table.rows[0].cells[0].firstChild.className = "state_blank";
        table.rows[0].cells[6].firstChild.className = "legal_stone";
    } else {
        table.rows[0].cells[0].firstChild.className = "state_blank";
        table.rows[0].cells[6].firstChild.className = "state_blank";
        game_end = true;
        end_game();
    }
    var table = document.getElementById("board");
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 2) {
                table.rows[y].cells[x].firstChild.innerText = "";
            }
        }
    }
    value_calced = false;
}

function ai_check() {
    if (game_end){
        clearInterval(ai_check);
    } else if (player == ai_player) {
        ai();
    } else if (show_value && !value_calced) {
        calc_value();
        value_calced = true;
    }
}

function draw(element){
    if (!element) { return; }
    var n = document.createTextNode(' ');
    var disp = element.style.display;
    element.appendChild(n);
    element.style.display = 'none';
    setTimeout(function(){
        element.style.display = disp;
        n.parentNode.removeChild(n);
    },20);
}

function empty(y, x) {
    return grid[y][x] == -1 || grid[y][x] == 2;
}

function inside(y, x) {
    return 0 <= y && y < hw && 0 <= x && x < hw;
}

function check_mobility() {
    var res = false;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (!empty(y, x))
                continue;
            grid[y][x] = -1;
            for (var dr = 0; dr < 8; ++dr) {
                var ny = y + dy[dr];
                var nx = x + dx[dr];
                if (!inside(ny, nx))
                    continue;
                if (empty(ny, nx))
                    continue;
                if (grid[ny][nx] == player)
                    continue;
                var flag = false;
                var nny = ny, nnx = nx;
                for (var d = 0; d < hw; ++d) {
                    if (!inside(nny, nnx))
                        break;
                    if (empty(nny, nnx))
                        break;
                    if (grid[nny][nnx] == player) {
                        flag = true;
                        break;
                    }
                    nny += dy[dr];
                    nnx += dx[dr];
                }
                if (flag) {
                    grid[y][x] = 2;
                    res = true;
                    break;
                }
            }
        }
    }
    return res;
}

async function ai() {
    let res = [
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1
    ];
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if(grid[y][x] == 0)
                res[y * hw + x] = 0;
            else if (grid[y][x] == 1)
                res[y * hw + x] = 1;
            else
                res[y * hw + x] = -1;
        }
    }
    var pointer = _malloc(hw2 * 4);
    var offset = pointer / 4;
    HEAP32.set(res, offset);
    var val = _ai_js(pointer, level_idx, ai_player);
    _free(pointer);
    console.log('val', val);
    var y = Math.floor(val / 1000 / hw);
    var x = Math.floor((val - y * 1000 * hw) / 1000);
    var dif_stones = val - y * 1000 * hw - x * 1000 - 100;
    console.log('y', y, 'x', x, 'dif_stones', dif_stones);
    move(y, x);
    update_graph(dif_stones);
}

function calc_value() {
    let res = new Int32Array([
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1, 
        -1, -1, -1, -1, -1, -1, -1, -1
    ]);
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if(grid[y][x] == 0)
                res[y * hw + x] = 0;
            else if (grid[y][x] == 1)
                res[y * hw + x] = 1;
            else
                res[y * hw + x] = -1;
        }
    }
    var n_byte = res.BYTES_PER_ELEMENT;
    var pointer_value = _malloc((hw2 + 10) * n_byte);
    var pointer = _malloc(hw2 * n_byte);
    HEAP32.set(res, pointer / n_byte);
    var hint_level = level_idx - 1;
    if (hint_level < 0){
        hint_level = 0;
    }
    if (hint_level > 7){
        hint_level = 7;
    }
    _calc_value(pointer, pointer_value, hint_level, ai_player);
    _free(pointer);
    var output_array = new Int32Array(HEAP32.buffer, pointer_value, hw2 + 10);
    _free(pointer_value);
    //console.log(output_array);
    var table = document.getElementById("board");
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 2) {
                if (-64 <= output_array[10 + y * hw + x] && output_array[10 + y * hw + x] <= 64){
                    table.rows[y].cells[x].firstChild.innerText = output_array[10 + y * hw + x];
                }
            }
        }
    }
}

function move(y, x) {
    for (var yy = 0; yy < hw; ++yy) {
        for (var xx = 0; xx < hw; ++xx) {
            bef_grid[yy][xx] = grid[yy][xx];
        }
    }
    grid[y][x] = player;
    for (var dr = 0; dr < 8; ++dr) {
        var ny = y + dy[dr];
        var nx = x + dx[dr];
        if (!inside(ny, nx))
            continue;
        if (empty(ny, nx))
            continue;
        if (grid[ny][nx] == player)
            continue;
        var flag = false;
        var nny = ny, nnx = nx;
        var plus = 0;
        for (var d = 0; d < hw; ++d) {
            if (!inside(nny, nnx))
                break;
            if (empty(nny, nnx))
                break;
            if (grid[nny][nnx] == player) {
                flag = true;
                break;
            }
            nny += dy[dr];
            nnx += dx[dr];
            ++plus;
        }
        if (flag) {
            for (var d = 0; d < plus; ++d) {
                grid[ny + d * dy[dr]][nx + d * dx[dr]] = player;
            }
        }
    }
    ++record.length;
    record[record.length - 1] = [y, x];
    update_record();
    ++n_stones;
    player = 1 - player;
    show(y, x);
}

function update_record() {
    var record_html = document.getElementById('record');
    var new_coord = String.fromCharCode(65 + record[record.length - 1][1]) + String.fromCharCode(49 + record[record.length - 1][0]);
    record_html.innerHTML += new_coord;
}

function update_graph(s) {
    if (show_graph){
        graph.data.labels.push(record.length);
        graph.data.datasets[0].data.push(s);
        graph.update();
    } else {
        let tmp = [record.length, s];
        graph_values.push(tmp);
    }
}

function end_game() {
    if (!show_graph){
        for (var i = 0; i < graph_values.length; ++i){
            graph.data.labels.push(graph_values[i][0]);
            graph.data.datasets[0].data.push(graph_values[i][1]);
        }
    }
    graph.update();
    let stones = [0, 0];
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (0 <= grid[y][x] <= 1) {
                ++stones[grid[y][x]];
            }
        }
    }
    html2canvas(document.getElementById('main'),{
        onrendered: function(canvas){
            var imgData = canvas.toDataURL();
            document.getElementById("game_result").src = imgData;
        }
    });
    var tweet_str = "";
    var hint = "without hints";
    if (show_value)
        hint = "with hints"
    if (stones[ai_player] < stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "You Win!";
        var dis = stones[1 - ai_player] - stones[ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = "I won against the world No.1 Othello AI " + level_names[level_idx] + " " + hint + " by " + dis + "discs! :)";
    } else if (stones[ai_player] > stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = "AI Win!";
        var dis = stones[ai_player] - stones[1 - ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = "I lose against the world No.1 Othello AI " + level_names[level_idx] + " " + hint + " by " + dis + "discs... :(";
    } else {
        document.getElementById('result_text').innerHTML = "Draw!";
        tweet_str = "I tied against the world No.1 Othello AI " + level_names[level_idx] + " " + hint + " by " + dis + "discs... :|";
    }
    var tweet_result = document.getElementById('tweet_result');
    tweet_result.innerHTML = 'Tweet this result!<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="' + tweet_str + '" data-url="https://www.egaroucid.nyanyan.dev/en/web/" data-hashtags="egaroucid" data-related="takuto_yamana" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>';
    twttr.widgets.load();
    var popup = document.getElementById('js-popup');
    if(!popup) return;
    popup.classList.add('is-show');
    var blackBg = document.getElementById('js-black-bg');
    tweet_result.classList.add('show');
    closePopUp(blackBg);
    function closePopUp(elem) {
        if(!elem) return;
        elem.addEventListener('click', function() {
            popup.classList.remove('is-show');
            tweet_result.classList.remove('show');
            tweet_result.innerHTML = "";
        })
    }
    document.getElementById('start').disabled = false;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = false;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = false;
    level_range.disabled = false;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i)
        players.item(i).disabled = false;
}
/*
var Module = {
    'noInitialRun' : false,
    'onRuntimeInitialized' : onruntimeinitialized
}

function onruntimeinitialized(){
    console.log("loaded AI");
    document.getElementById('start').value = "対局開始";
    document.getElementById('start').disabled = false;
}
*/
window.onload = function() {
    level_range.addEventListener('input', rangeOnChange);
    setCurrentValue(level_range.value);
    var container = document.getElementById('chart_container');
    ctx.clientWidth = container.clientWidth;
    ctx.clientHeight = container.clientHeight;
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    var coord_top = document.getElementById('coord_top');
    var row = document.createElement('tr');
    for (var x = 0; x < hw; ++x) {
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        coord.innerHTML = String.fromCharCode(65 + x);
        cell.appendChild(coord);
        row.appendChild(cell);
    }
    coord_top.appendChild(row);
    var coord_left = document.getElementById('coord_left');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        coord.innerHTML = y + 1;
        cell.appendChild(coord);
        row.appendChild(cell);
        coord_left.appendChild(row);
    }
    var coord_right = document.getElementById('coord_right');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        var cell = document.createElement('td');
        cell.className = "coord_cell";
        var coord = document.createElement('span');
        coord.className = "coord";
        cell.appendChild(coord);
        row.appendChild(cell);
        coord_right.appendChild(row);
    }
    var table = document.getElementById('board');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        for (var x = 0; x < hw; ++x) {
            var cell = document.createElement('td');
            cell.className = "cell";
            var stone = document.createElement('span');
            stone.className = "empty_stone";
            cell.appendChild(stone);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    show(-2, -2);
    console.log("loading AI");
    //document.getElementById('start').value = "AI読込中";
    document.getElementById('start').disabled = true;
    /*
    var worker = new Worker("assets/js/init_worker.js");
    worker.addEventListener('message', function(e) {
        console.log('Worker said: ', e.data);
        ai_initialized = e.data;
        console.log("loaded AI");
        document.getElementById('start').value = "対局開始";
        document.getElementById('start').disabled = false;
    }, false);
    worker.postMessage('init');
    */
    setInterval(try_initialize_ai, 250);
    //ai_init_p();
    //setInterval(check_initialized, 250);
};

function try_initialize_ai(){
    if (document.getElementById('start').value == 'AI Initializing'){
        try{
            _init_ai();
            console.log("loaded AI");
            document.getElementById('start').value = "Start";
            document.getElementById('start').disabled = false;
            document.getElementById('reset').disabled = false;
        } catch(exception){
            console.error(exception);
            document.getElementById('start').value = "Failed Initializing Please Reload";
            document.getElementById('start').disabled = true;
        }
        clearInterval(try_initialize_ai);
    }
}

function reset(){
    document.getElementById('start').disabled = false;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = false;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = false;
    level_range.disabled = false;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i){
        players.item(i).disabled = false;
    }
    for (var y = 0; y < hw; ++y){
        for (var x = 0; x < hw; ++x) {
            grid[y][x] = -1;
            bef_grid[y][x] = -1;
        }
    }
    graph.data.labels = [];
    graph.data.datasets[0].data = [];
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    graph.data.values = [];
    graph.data.labels = [];
    graph.update();
    game_end= true;
    show(-1, -1);
}