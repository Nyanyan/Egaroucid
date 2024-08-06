# Egaroucid for Web

<div align="center">
    <span>これは[ダウンロード版](./../download/)よりも弱い簡易バージョンです。機能も非常に少ないです。豊富な機能の全てを使いたい場合はダウンロード版をお使いください。<br>
    10秒経っても読み込めない場合はリロードしてください。<br></span>
    <span id="ai_info">AI読み込み中…</span>
</div>
<div align="center">
    <table class="settings_table">
        <tr class="settings_table">
            <td class="settings_table"><label><input type="radio" class="radio_size" name="ai_player" value="1" id="white" checked><span class="setting">黒(先手): あなた 白(後手): AI</span></label></td>
        </tr>
        <tr class="settings_table">
            <td class="settings_table"><label><input type="radio" class="radio_size" name="ai_player" value="0" id="black"><span class="setting">黒(先手): AI 白(後手): あなた</span></label></td>
        </tr>
    </table>
</div>
<div align="center">
    <label>
        <span class="setting">AIの強さ</span>
        <input type="range" id="ai_level" min="0" max="15" step="1" value="7">
    </label>
    <span class="setting" id="ai_level_label"></span>
</div>
<div align="center">
    <table class="settings_table">
        <tr class="settings_table">
            <td class="settings_table"><label><input type="checkbox" id="show_value" unchecked><span class="setting">ヒント</span></label></td>
            <td class="settings_table"><label><input type="checkbox" id="show_graph" checked><span class="setting">グラフ</span></label></td>
        </tr>
        <tr class="settings_table">
            <td class="settings_table"><label><input type="checkbox" id="show_legal" checked><span class="setting">合法手</span></label></td>
            <td class="settings_table"><label><input type="checkbox" id="auto_pass" checked><span class="setting">自動パス</span></label></td>
        </tr>
    </table>
</div>
<div align="center">
    <div class="buttons">
        <input type="submit" class="setting" value="対局開始" onclick="start()" id="start" disabled>
        <input type="submit" class="setting" value="リセット" onclick="reset()" id="reset" disabled>
        <input type="submit" class="setting" value="パス" onclick="pass()" id="pass" disabled>
    </div>
</div>
<div class="popup" id="js-popup">
    <div class="popup-inner">
        <p align="center" class="sub_title" id="result_text"></p>
        <img class="image" id="game_result" width="80%" height="auto" style="aspect-ratio: 1.06;">
        <p align="center" class="hidden" id="tweet_result"></p>
        <p align="center" class="text">上の画像は右クリックなどで保存できます。</p>
        <p align="center" class="text">周りをタップするとポップアップが消えます。</p>
    </div>
    <div class="black-background" id="js-black-bg"></div>
</div>
<div class="div_main">
    <table class="table_board" align="center" id="table_board"></table>
    <table class="status" id="status" align="center">
        <tr>
            <td class="status_cell"><span class="state_blank"></span></td>
            <td class="status_cell"><span class="black_stone"></span></td>
            <td class="status_char"><span class="state_blank">2</span></td>
            <td class="status_char"><span class="state_blank">-</span></td>
            <td class="status_char"><span class="state_blank">2</span></td>
            <td class="status_cell"><span class="white_stone"></span></td>
            <td class="status_cell"><span class="state_blank"></span></td>
        </tr>
    </table>
</div>
<div id="info" class="info">
    <div class="sub_title">対局情報</div>
    <div class="sub_sub_title">予想最終石差グラフ</div>
    <div class="chart" id="chart_container">
        <canvas id="graph"></canvas>
    </div>
    <div class="sub_sub_title">棋譜</div>
    <div class="record" id="record"></div>
</div>
<div align="center">
    <div class="sub_title">使い方</div>
    <div class="text">
        手番とAIの強さを選択し、対局開始ボタンを押してください。<br>
        予想最終石差グラフは左が序盤、右が直近の手です。値が大きくなるほどAI有利と判断していて、0が互角です。<br>
        グラフ表示をオフにしても対局終了後にグラフが描画されます。<br>
        ヒントは最大でも7手読みの評価値ですので精度は悪めです。<br>
    </div>
    <p></p>
    <div class="sub_title">AIの強さ</div>
    <div class="text">
        AIの強さは中盤の先読み手数、終盤の読み切り手数、及び完全読み手数で調整されます。
        レベルが上がると計算時間が増えるので、様子を見つつ設定してください。<br>
        各強さの詳細は以下の通りです。
    </div>
    <div class="table_wrapper">
        <table>
            <tr>
                <th class="text">レベル</th>
                <th class="text">中盤読み</th>
                <th class="text">終盤読み切り</th>
                <th class="text">完全読み</th>
            </tr>
            <tr>
                <td class="text">0</td>
                <td class="text">0手</td>
                <td class="text">0手</td>
                <td class="text">0手</td>
            </tr>
            <tr>
                <td class="text">1</td>
                <td class="text">1手</td>
                <td class="text">2手</td>
                <td class="text">2手</td>
            </tr>
            <tr>
                <td class="text">2</td>
                <td class="text">2手</td>
                <td class="text">4手</td>
                <td class="text">4手</td>
            </tr>
            <tr>
                <td class="text">3</td>
                <td class="text">3手</td>
                <td class="text">6手</td>
                <td class="text">6手</td>
            </tr>
            <tr>
                <td class="text">4</td>
                <td class="text">4手</td>
                <td class="text">8手</td>
                <td class="text">8手</td>
            </tr>
            <tr>
                <td class="text">5</td>
                <td class="text">5手</td>
                <td class="text">10手</td>
                <td class="text">10手</td>
            </tr>
            <tr>
                <td class="text">6</td>
                <td class="text">6手</td>
                <td class="text">12手</td>
                <td class="text">12手</td>
            </tr>
            <tr>
                <td class="text">7</td>
                <td class="text">7手</td>
                <td class="text">14手</td>
                <td class="text">14手</td>
            </tr>
            <tr>
                <td class="text">8</td>
                <td class="text">8手</td>
                <td class="text">16手</td>
                <td class="text">16手</td>
            </tr>
            <tr>
                <td class="text">9</td>
                <td class="text">9手</td>
                <td class="text">18手</td>
                <td class="text">18手</td>
            </tr>
            <tr>
                <td class="text">10</td>
                <td class="text">10手</td>
                <td class="text">20手</td>
                <td class="text">20手</td>
            </tr>
            <tr>
                <td class="text">11</td>
                <td class="text">11手</td>
                <td class="text">20手</td>
                <td class="text">20手</td>
            </tr>
            <tr>
                <td class="text">12</td>
                <td class="text">12手</td>
                <td class="text">22手</td>
                <td class="text">20手</td>
            </tr>
            <tr>
                <td class="text">13</td>
                <td class="text">13手</td>
                <td class="text">22手</td>
                <td class="text">20手</td>
            </tr>
            <tr>
                <td class="text">14</td>
                <td class="text">14手</td>
                <td class="text">24手</td>
                <td class="text">22手</td>
            </tr>
            <tr>
                <td class="text">15</td>
                <td class="text">15手</td>
                <td class="text">24手</td>
                <td class="text">22手</td>
            </tr>
        </table>
    </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.bundle.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.4.1/html2canvas.js"></script>
<script src="script.js"></script>

