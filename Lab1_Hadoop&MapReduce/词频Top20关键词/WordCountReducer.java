import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    // 用于存储单词及其出现次数的映射
    private Map<String, Integer> wordCounts = new HashMap<>();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws
            IOException, InterruptedException {
        // 1. 累加求和
        int sum = 0;
        for (IntWritable count : values) {
            sum += count.get();
        }
        // 2. 将结果存入 Map
        wordCounts.put(key.toString(), sum);
    }
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        // 1. 将 Map 转换为 List 以便排序
        List<Map.Entry<String, Integer>> entryList = new ArrayList<>(wordCounts.entrySet());
        // 2. 按照出现次数从高到低排序
        Collections.sort(entryList, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> e1, Map.Entry<String, Integer> e2) {
                return e2.getValue().compareTo(e1.getValue());
            }
        });
        // 3. 输出出现频率最高的 20 个关键词
        int count = 0;
        for (Map.Entry<String, Integer> entry : entryList) {
            if (count >= 20) {
                break;
            }
            context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            count++;
        }
    }
}
