import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable>{
    IntWritable v = new IntWritable();
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        // 1 找到最大值
        int maxScore = Integer.MIN_VALUE;
        for (IntWritable value : values) {
            maxScore = Math.max(maxScore, value.get());
        }
        // 2 输出
        v.set(maxScore);
        context.write(key, v);
    }
}
