<?php


function writeData($data) {
    $myfile = "f1.csv";
    $fh = fopen($myfile, 'a');
    if ( $fh == false ) {
        echo( "Error opening file" );
        exit();
    }
    fwrite($fh, $data);
    fwrite($fh, "\n");
    fclose($fh);
}


if ($_POST["t"] && $_POST["x"] && $_POST["y"]) {
    $csv = sprintf('%d, %d, %d', $_POST["t"], $_POST["x"], $_POST["y"]);
    writeData(str_replace('.', '', $csv));
} else {
    echo "Error pasring request";
    exit();
}

?>


