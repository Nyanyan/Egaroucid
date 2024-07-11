const lang_level = 'レベル';
const lang_expected_score = '予想最終石差';
const lang_without_hint = 'ヒントなし';
const lang_with_hint = 'ヒントあり';
const lang_you_win = 'あなたの勝ち！';
const lang_ai_win = 'AIの勝ち！';
const lang_draw = '引き分け！';
const lang_tweet_str_0_win = '';
const lang_tweet_str_0_lose = '';
const lang_tweet_str_0_draw = '';
const lang_tweet_str_1 = 'オセロAI Egaroucid for Web ';
const lang_tweet_str_2 = '';
const lang_tweet_str_3 = 'に';
const lang_tweet_str_4 = '';
const lang_tweet_str_5_win = '石勝ちしました！';
const lang_tweet_str_5_lose = '石負けしました…';
const lang_tweet_str_5_draw = 'と引き分けました！';
const lang_tweet_result = '結果をツイート！';
const lang_ai_loading = 'AI読み込み中…\n10秒経っても読み込めない場合はリロードしてください';
const lang_ai_loaded = 'AI読み込み完了！';
const lang_ai_load_failed = 'AI読み込み失敗 リロードしてください';



let hw = 8;
let hw2 = 64;
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
let n_stones = 4;
let player = 0;
let ai_player = -1;
let level_idx = 0;
let level_names = [];
let game_end = false;
let value_calculated = false;
let record = [];
let step = 0;
let isstart = true;
let show_value = true;
let show_graph = true;
let show_legal = true;
let auto_pass = true;
let ai_initializing = true;
let graph_values = [];
let ctx = document.getElementById("graph");
let loading_percent = 0;
let percent_pointer = null;
let initializing_var = null;
let graph = new Chart(ctx, {
    type: 'line',
    data: {
    labels: [],
    datasets: [
        {
        label: lang_expected_score,
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

var Module = {
    'noInitialRun' : true,
    'onRuntimeInitialized' : initialize_ai_wrapper
}

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
    var show_legal_elem = document.getElementById('show_legal');
    show_legal_elem.disabled = true;
    show_legal = show_legal_elem.checked;
    var auto_pass_elem = document.getElementById('auto_pass');
    auto_pass_elem.disabled = true;
    auto_pass = auto_pass_elem.checked;
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

function pass(){
    var pass_elem = document.getElementById('pass');
    pass_elem.disabled = true;
    player = 1 - player;
}

function show(r, c) {
    console.log(r, c);
    var table = document.getElementById("board");
    var waiting_pass = false;
    if (!check_mobility()) {
        player = 1 - player;
        if (!check_mobility()) {
            player = 2;
        } else if (!auto_pass && player == ai_player){
            player = 1 - player;
            var pass_elem = document.getElementById('pass');
            pass_elem.disabled = false;
            waiting_pass = true;
        }
    }
    for (var y = 0; y < 8; ++y) {
        for (var x = 0; x < 8; ++x) {
            var stone_id = "stone_" + (y * hw + x);
            var cell_id = "cell_" + (y * hw + x);
            document.getElementById(cell_id).style.backgroundColor = "#249972";
            document.getElementById(stone_id).innerHTML = "";
            if (grid[y][x] == 0) {
                if (bef_grid[y][x] != 0) {
                    document.getElementById(stone_id).className = "black_stone";
                    document.getElementById(cell_id).setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 1) {
                if (bef_grid[y][x] != 1) {
                    document.getElementById(stone_id).className = "white_stone";
                    document.getElementById(cell_id).setAttribute('onclick', "");
                }
            } else if (grid[y][x] == 2 && !waiting_pass) {
                if (r == -1 || inside(r, c)) {
                    if (show_legal) {
                        if (player == 0) {
                            document.getElementById(stone_id).className = "legal_stone_black";
                        } else {
                            document.getElementById(stone_id).className = "legal_stone_white";
                        }
                    } else {
                        document.getElementById(stone_id).className = "empty_stone";
                    }
                    document.getElementById(cell_id).setAttribute('onclick', "move(" + y + ", " + x + ")");
                } else {
                    document.getElementById(stone_id).className = "empty_stone";
                    document.getElementById(cell_id).setAttribute('onclick', "");
                }
            } else {
                document.getElementById(stone_id).className = "empty_stone";
                document.getElementById(cell_id).setAttribute('onclick', "");
            }
        }
    }
    if (inside(r, c)) {
        var cell_id = "cell_" + (r * hw + c);
        document.getElementById(cell_id).style.backgroundColor = "#d14141";
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
    value_calculated = false;
}

function ai_check() {
    if (game_end){
        clearInterval(ai_check);
    } else if (player == ai_player) {
        ai();
    } else if (show_value && !value_calculated) {
        calc_value();
        value_calculated = true;
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
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 2) {
                if (-64 <= output_array[10 + y * hw + x] && output_array[10 + y * hw + x] <= 64){
                    var stone_id = "stone_" + (y * hw + x);
                    document.getElementById(stone_id).innerHTML = output_array[10 + y * hw + x];
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
    var new_coord = String.fromCharCode(97 + record[record.length - 1][1]) + String.fromCharCode(49 + record[record.length - 1][0]);
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
    html2canvas(document.getElementById('table_board'),{
        onrendered: function(canvas){
            var imgData = canvas.toDataURL();
            document.getElementById("game_result").src = imgData;
        }
    });
    var tweet_str = "";
    var hint = lang_without_hint;
    if (show_value)
        hint = lang_with_hint;
    if (stones[ai_player] < stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = lang_you_win;
        var dis = stones[1 - ai_player] - stones[ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = lang_tweet_str_0_win + lang_tweet_str_1 + level_names[level_idx] + lang_tweet_str_2 + hint + lang_tweet_str_3 + dis + lang_tweet_str_4 + lang_tweet_str_5_win;
    } else if (stones[ai_player] > stones[1 - ai_player]) {
        document.getElementById('result_text').innerHTML = lang_ai_win;
        var dis = stones[ai_player] - stones[1 - ai_player] + hw2 - stones[ai_player] - stones[1 - ai_player];
        tweet_str = lang_tweet_str_0_lose + lang_tweet_str_1 + level_names[level_idx] + lang_tweet_str_2 + hint + lang_tweet_str_3 + dis + lang_tweet_str_4 + lang_tweet_str_5_lose;
    } else {
        document.getElementById('result_text').innerHTML = lang_draw;
        tweet_str = lang_tweet_str_0_draw + lang_tweet_str_1 + level_names[level_idx] + lang_tweet_str_2 + hint + lang_tweet_str_4 + lang_tweet_str_5_draw;
    }
    var tweet_result = document.getElementById('tweet_result');
    tweet_result.innerHTML = lang_tweet_result + '<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="' + tweet_str + '" data-url="https://www.egaroucid.nyanyan.dev/ja/web/" data-hashtags="egaroucid" data-related="takuto_yamana" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>';
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
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = false;
    var show_legal_elem = document.getElementById('show_legal');
    show_legal_elem.disabled = false;
    var auto_pass_elem = document.getElementById('auto_pass');
    auto_pass_elem.disabled = false;
    level_range.disabled = false;
    let players = document.getElementsByName('ai_player');
    for (var i = 0; i < 2; ++i)
        players.item(i).disabled = false;
}

window.onload = function() {
    for (var i = 0; i <= 15; ++i) {
        level_names.push(lang_level + i);
    }
    level_names
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
    var table = document.getElementById('table_board');
    for (var y = -1; y < hw; ++y) {
        var row = document.createElement('tr');
        for (var x = -1; x < hw + 1; ++x) {
            var cell = document.createElement('td');
            var stone_elem = document.createElement('span');
            if (y == -1) {
                cell.className = "coord_cell";
                stone_elem.className = "coord";
                if (0 <= x && x < hw) {
                    stone_elem.innerHTML = "abcdefgh"[x];
                }
            } else if (x == -1 || x == hw) {
                cell.className = "coord_cell";
                stone_elem.className = "coord";
                if (x == -1) {
                    stone_elem.innerHTML = y;
                }
            } else {
                cell.className = "cell";
                cell.id = "cell_" + (y * hw + x);
                stone_elem.className = "empty_stone";
                stone_elem.id = "stone_" + (y * hw + x);
            }
            cell.appendChild(stone_elem);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    show(-2, -2);

    //document.getElementById('ai_info').innerText = lang_ai_loading + " " + loading_percent + "%";
    document.getElementById('ai_info').innerText = lang_ai_loading;

    const scriptElem = document.createElement('script');
    scriptElem.src = 'ai.js';
    //scriptElem.addEventListener('load', (e) => {
    //document.getElementById('ai_info').innerText = lang_ai_initializing;
    //});
    document.body.appendChild(scriptElem);

    //setInterval(display_loading, 100);
};

/*
function display_loading(){
    console.log('ptr', percent_pointer);
    if (percent_pointer != null){
        loading_percent = new Int32Array(HEAP32.buffer, percent_pointer, 1)[0];
        console.log(loading_percent);
        document.getElementById('ai_info').innerText = lang_ai_loading + " " + loading_percent + "%";
    }
}
    */

function initialize_ai_wrapper(){
    initializing_var = setInterval(initialize_ai, 100);
}

function initialize_ai(){
    if (Module['onRuntimeInitialized']){
        try{
            percent_pointer = _malloc(4);
            var init_result = _init_ai(percent_pointer);
            loading_percent = new Int32Array(HEAP32.buffer, percent_pointer, 1)[0];
            _free(percent_pointer);
            percent_pointer = null;
            if (init_result == 0){
                console.log("loaded AI");
                document.getElementById('start').disabled = false;
                document.getElementById('reset').disabled = false;
                ai_initializing = false;
                document.getElementById('ai_info').innerText = lang_ai_loaded;
            } else{
                console.error(exception);
                document.getElementById('ai_info').innerText = lang_ai_load_failed;
            }
        } catch(exception){
            console.error(exception);
            document.getElementById('ai_info').innerText = lang_ai_load_failed;
        }
        clearInterval(initializing_var);
    }
}

function reset(){
    document.getElementById('start').disabled = false;
    var show_value_elem = document.getElementById('show_value');
    show_value_elem.disabled = false;
    show_value = show_value_elem.checked;
    var show_graph_elem = document.getElementById('show_graph');
    show_graph_elem.disabled = false;
    var show_legal_elem = document.getElementById('show_legal');
    show_legal_elem.disabled = false;
    var auto_pass_elem = document.getElementById('auto_pass');
    auto_pass_elem.disabled = false;
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
    show(-2, -2);
}