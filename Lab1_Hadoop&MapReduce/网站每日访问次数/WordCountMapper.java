import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable>{
    Text k = new Text();
    IntWritable v = new IntWritable(1);
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // 1. 获取一行
        String line = value.toString();

        // 2. 切割，CSV 文件以逗号作为分隔符
        String[] fields = line.split(",");

        // 3. 判断是否有足够的字段（至少有5列）
        if(fields.length >= 5){
            // 4. 获取访问日期和时间（第5列）
            String dateTime = fields[4].trim();
            // 5. 从日期和时间中提取日期部分
            String[] dateTimeParts = dateTime.split("\\s+");
            if(dateTimeParts.length >= 1){
                String date = dateTimeParts[0]; // 提取日期部分
                // 6. 输出键值对，日期作为键，次数为1
                k.set(date);
                context.write(k, v);
            }
        }
    }
}
